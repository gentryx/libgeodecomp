// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/io/logger.h>
#include <cfloat>

#define LIBGEODECOMP_DEBUG_LEVEL 4
namespace LibGeoDecomp{

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
class AutoTuningSimulator
{
public:
    typedef boost::shared_ptr<SimulationFactory<CELL_TYPE> > SimFactoryPtr;
    class Simulation{
    public:
        Simulation(std::string name,
            SimFactoryPtr simFactory,
            SimulationParameters param,
            double fit = DBL_MAX):
            simulationType(name),
            simulationFactory(simFactory),
            parameters(param),
            fitness(fit)
        {}
        std::string simulationType;
        SimFactoryPtr simulationFactory;
        SimulationParameters parameters;
        double fitness;
    };

    typedef boost::shared_ptr<Simulation> SimulationPtr;

    AutoTuningSimulator(Initializer<CELL_TYPE> *initializer);

    ~AutoTuningSimulator() {}

    void addWriter(ParallelWriter<CELL_TYPE> *writer);

    void addWriter(Writer<CELL_TYPE> *writer);

    void addSteerer(const Steerer<CELL_TYPE> *steerer);

    void addNewSimulation(std::string name, std::string typeOfSimulation);

    void deleteAllSimulations();

    std::vector<std::string> getSimulationNames();

    std::string getSimulatorType(std::string simulationName);

    double getFitness(std::string simulationName);

    SimulationParameters getSimulationParameters(std::string simulationName);

    void setSimulationSteps(unsigned steps);

    void setParameters(SimulationParameters params, std::string name);

    void run();

    std::string getBestSim();

    void runToCompletion(std::string optimizerName);

    unsigned normalizeSteps(double goal, unsigned start = defaultInitializerSteps);

    void runTest();

    void prepareSimulations();

private:
    void addNewSimulation(std::string name,
        std::string typeOfSimulation,
        VarStepInitializerProxy<CELL_TYPE> &initializer);

    bool isInMap(const std::string name)const;

    std::map<const std::string, SimulationPtr> simulations;
    unsigned optimizationSteps; // maximum number of Steps for the optimizer
    VarStepInitializerProxy<CELL_TYPE> varStepInitializer;
    std::vector<boost::shared_ptr<ParallelWriter<CELL_TYPE> > > parallelWriters;
    std::vector<boost::shared_ptr<Writer<CELL_TYPE> > > writers;
    std::vector<boost::shared_ptr<Steerer<CELL_TYPE> > > steerers;
 
    /**
     * If some missconfiguration is happned the defaultInitializerSteps will be used.
     */
    static const unsigned defaultInitializerSteps = 5;
}; //AutoTunigSimulator

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::AutoTuningSimulator(Initializer<CELL_TYPE> *initializer):
    optimizationSteps(10),
    varStepInitializer(initializer)
{
    addNewSimulation("SerialSimulation",
        "SerialSimulation",
        varStepInitializer);

#ifdef LIBGEODECOMP_WITH_THREADS
    addNewSimulation("CacheBlockingSimulation",
        "CacheBlockingSimulation",
        varStepInitializer);
#endif

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
    addNewSimulation("CudaSimulation",
        "CudaSimulation",
        varStepInitializer);
#endif
#endif
}


template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addWriter(ParallelWriter<CELL_TYPE> *writer)
{
    parallelWriters.push_back(boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer));
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addWriter(Writer<CELL_TYPE> *writer)
{
    writers.push_back(boost::shared_ptr<Writer<CELL_TYPE> >(writer));
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addSteerer(const Steerer<CELL_TYPE> *steerer)
{
    steerers.push_back(boost::shared_ptr<Steerer<CELL_TYPE> >(steerer));
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addNewSimulation(std::string name, std::string typeOfSimulation)
{
    addNewSimulation(name, typeOfSimulation, varStepInitializer);
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addNewSimulation(std::string name,
    std::string typeOfSimulation,
    VarStepInitializerProxy<CELL_TYPE> &initializer)
{
    if (typeOfSimulation == "SerialSimulation") {
        SimFactoryPtr simFac_p(new SerialSimulationFactory<CELL_TYPE>(&initializer));
        SimulationPtr sim_p(new Simulation(
                typeOfSimulation,
                simFac_p,
                simFac_p->parameters()));
        simulations[name] = sim_p;
        return;
    }

#ifdef LIBGEODECOMP_WITH_THREADS
    if (typeOfSimulation == "CacheBlockingSimulation") {
        SimFactoryPtr simFac_p(new CacheBlockingSimulationFactory<CELL_TYPE>(&initializer));
        SimulationPtr sim_p(new Simulation(
                typeOfSimulation,
                simFac_p,
                simFac_p->parameters()));
        simulations[name] = sim_p;
        return;
    }
#endif

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
     if (typeOfSimulation == "CudaSimulation") {
        SimFactoryPtr simFac_p(new CudaSimulationFactory<CELL_TYPE>(&initializer));
        SimulationPtr sim_p(new Simulation(
                typeOfSimulation,
                simFac_p,
                simFac_p->parameters()));
        simulations[name] = sim_p;
        return;
     }
#endif
#endif

    throw std::invalid_argument("SimulationFactory::addNewSimulation(): unknown simulator type");
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::deleteAllSimulations()
{
    simulations.clear();
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
std::vector<std::string> AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getSimulationNames()
{
    std::vector<std::string> result;
    typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
    for (IterType iter = simulations.begin(); iter != simulations.end(); iter++)
    {
        result.push_back(iter->first);
    }
    return result;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
std::string AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getSimulatorType(std::string simulationName)
{
    if (isInMap(simulationName)){
        return simulations[simulationName]->simulationType;
    } else {
        throw std::invalid_argument("getSimulatorType(simulationName)) get invalid simulationName");
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
double AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getFitness(std::string simulationName)
{
    if (isInMap(simulationName)){
        return simulations[simulationName]->fitness;
    } else {
        throw std::invalid_argument("getFitness(simulationName) get invalid simulationName");
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
SimulationParameters AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getSimulationParameters(std::string simulationName)
{
    if (isInMap(simulationName)){
        return simulations[simulationName]->parameters;
    } else {
        throw std::invalid_argument("getSimulationParameters(simulationName) get invalid simulationName");
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::setSimulationSteps(unsigned steps)
{
    optimizationSteps = steps;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::setParameters(SimulationParameters params, std::string name)
{
    if (isInMap(name)) {
        simulations[name]->parameters = params;
    } else {
        throw std::invalid_argument(
            "AutotuningSimulatro<...>::setParameters(params,name) get invalid name");
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::run()
{
    prepareSimulations();
    if(normalizeSteps(-0.25)){
        runTest();
    }else{
        LOG(Logger::WARN,"normalize Steps was not successful, a default value is used!")
        varStepInitializer.setMaxSteps(5);
        runTest();
    }
    std::string best = getBestSim();
    runToCompletion(best);
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
std::string AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getBestSim()
{
std::string bestSimulation;
double tmpFitness = DBL_MAX;
typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
for (IterType iter = simulations.begin(); iter != simulations.end(); iter++) {
    if (iter->second->fitness < tmpFitness){
        tmpFitness = iter->second->fitness;
        bestSimulation = iter->first;
    }
}
LOG(Logger::DBG, "bestSimulation: " << bestSimulation)
return bestSimulation;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::runToCompletion(std::string optimizerName)
{
    if ( ! isInMap(optimizerName))
        throw std::invalid_argument(
            "AutotuningSimulator<...>::runToCompletion() get invalid optimizerName");
    shared_ptr<Initializer<CELL_TYPE> > originalInitializer = varStepInitializer.getInitializer();
    varStepInitializer.setMaxSteps(originalInitializer->maxSteps());
    simulations[optimizerName]->simulationFactory->operator()(simulations[optimizerName]->parameters);
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
unsigned AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::normalizeSteps(double goal, unsigned start)
{
    LOG(Logger::INFO, "normalizeSteps")
    if (! isInMap("SerialSimulation")){
        // FIXME Maybe an exception can be the heavy way!
        LOG(Logger::WARN, "AutoTuningSimulator::normalizeSteps(): no "
                    << "SerialSimulation available!")
        return 0;
    }
    if (start == 0) {
        LOG(Logger::WARN, "normalizeSteps is called with start = 0, "
                    << "this set start to default!")
        start = defaultInitializerSteps;
    }
    SimFactoryPtr factory = simulations["SerialSimulation"]->simulationFactory;
    unsigned steps = start;
    unsigned oldSteps = start;
    varStepInitializer.setMaxSteps(1);
    double variance = factory->operator()(simulations["SerialSimulation"]->parameters);
    double fitness = DBL_MAX;
    do {
        varStepInitializer.setMaxSteps(steps);
        fitness = factory->operator()(simulations["SerialSimulation"]->parameters);
        oldSteps = steps;
        LOG(Logger::DBG, "steps: " << steps)
        steps = ((double) steps / fitness) * (double)goal;
        if (steps < 1){
            steps =1;
        }
        LOG(Logger::DBG, "new calculated steps: " << steps)
        LOG(Logger::DBG, "fitness: " << fitness << " goal: " << goal)
    } while((!(fitness > goal + variance && fitness < goal - variance ))
         && (!(oldSteps <= 1 && fitness > goal)));

    return oldSteps;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::runTest()
{
    typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
    for (IterType iter = simulations.begin(); iter != simulations.end(); iter++) {
        OPTIMIZER_TYPE optimizer(iter->second->parameters);
        iter->second->parameters = optimizer(
            optimizationSteps,
            *iter->second->simulationFactory);
        iter->second->fitness = optimizer.getFitness();

        LOG(Logger::DBG, "Result of the " << iter->second->simulationType
            << ": " << iter->second->fitness << std::endl
            << "new Parameters:"<< std::endl << iter->second->parameters
            << std::endl);
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::prepareSimulations()
{
    typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
    for (IterType iter = simulations.begin(); iter != simulations.end(); iter++) {
        LOG(Logger::DBG, iter->first);
        for (unsigned i = 0; i < writers.size(); ++i) {
            iter->second->simulationFactory->addWriter(*writers[i].get()->clone());
        }
        for (unsigned i = 0; i < parallelWriters.size(); ++i) {
            iter->second->simulationFactory->addWriter(*parallelWriters[i].get()->clone());
        }
        for (unsigned i = 0; i < steerers.size(); ++i) {
            iter->second->simulationFactory->addSteerer(*steerers[i].get()->clone());
        }
    }
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
    bool AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::isInMap(const std::string name)const
{
    if (simulations.find(name) == simulations.end()){
        return false;
    } else {
        return true;
    }
}
} // namespace LibGeoDecomp

#endif
