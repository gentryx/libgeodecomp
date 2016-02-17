// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/io/logger.h>
#include <cfloat>

namespace LibGeoDecomp {

namespace AutoTuningSimulatorHelpers {

template<typename CELL_TYPE>
class Simulation{
public:
    typedef boost::shared_ptr<SimulationFactory<CELL_TYPE> > SimFactoryPtr;

    Simulation(
        const std::string& name,
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

}

/**
 * This Simulator makes use of LibGeoDecomp's parameter optimization
 * facilities to select the most efficient Simulator implementation
 * and suitable parameters for the given simulation model and
 * hardware.
 */
template<typename CELL_TYPE, typename OPTIMIZER_TYPE>
class AutoTuningSimulator
{
public:
    friend class AutotuningSimulatorTest;
    friend class AutotuningSimulatorWithCudaTest;

    typedef AutoTuningSimulatorHelpers::Simulation<CELL_TYPE>  Simulation;
    typedef boost::shared_ptr<SimulationFactory<CELL_TYPE> > SimFactoryPtr;
    typedef boost::shared_ptr<Simulation> SimulationPtr;

    AutoTuningSimulator(Initializer<CELL_TYPE> *initializer, unsigned optimizationSteps = 10);

    void addWriter(ParallelWriter<CELL_TYPE> *writer);

    void addWriter(Writer<CELL_TYPE> *writer);

    void addSteerer(const Steerer<CELL_TYPE> *steerer);

    void run();

private:
    std::map<const std::string, SimulationPtr> simulations;
    unsigned optimizationSteps; // maximum number of Steps for the optimizer
    boost::shared_ptr<VarStepInitializerProxy<CELL_TYPE> > varStepInitializer;
    std::vector<boost::shared_ptr<ParallelWriter<CELL_TYPE> > > parallelWriters;
    std::vector<boost::shared_ptr<Writer<CELL_TYPE> > > writers;
    std::vector<boost::shared_ptr<Steerer<CELL_TYPE> > > steerers;

    // fixme: why pass a string here and not a SimFactory instance? also: parameters and SimFactory should be added simultaneously
    void addNewSimulation(const std::string& name, const std::string& typeOfSimulation);
    void setParameters(const SimulationParameters& params, const std::string& name);

    std::string getBestSim();

    void runToCompletion(const std::string& optimizerName);

    unsigned normalizeSteps(double goal, unsigned startStepNum);

    void runTest();

    void prepareSimulations();

    SimulationPtr getSimulation(const std::string& simulatorName)
    {
        if (simulations.find(simulatorName) == simulations.end()) {
            throw std::invalid_argument("AutoTuningSimulator could not find simulatorName in registry of factories");
        }

        return simulations[simulatorName];
    }
};

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::AutoTuningSimulator(Initializer<CELL_TYPE> *initializer, unsigned optimizationSteps):
    optimizationSteps(optimizationSteps),
    varStepInitializer(new VarStepInitializerProxy<CELL_TYPE>(initializer))
{
    addNewSimulation("SerialSimulation", "SerialSimulation");

#ifdef LIBGEODECOMP_WITH_THREADS
    addNewSimulation("CacheBlockingSimulation", "CacheBlockingSimulation");
#endif

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
    addNewSimulation("CudaSimulation", "CudaSimulation");
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
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addNewSimulation(
    const std::string& name,
    const std::string& typeOfSimulation)
{
    // fixme: if we already have specialized factories, we should not have this additional manual type switch
    if (typeOfSimulation == "SerialSimulation") {
        SimFactoryPtr simFac_p(new SerialSimulationFactory<CELL_TYPE>(varStepInitializer));
        SimulationPtr sim_p(new Simulation(
                typeOfSimulation,
                simFac_p,
                simFac_p->parameters()));
        simulations[name] = sim_p;
        return;
    }

#ifdef LIBGEODECOMP_WITH_THREADS
    if (typeOfSimulation == "CacheBlockingSimulation") {
        SimFactoryPtr simFac_p(new CacheBlockingSimulationFactory<CELL_TYPE>(varStepInitializer));
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
        SimFactoryPtr simFac_p(new CudaSimulationFactory<CELL_TYPE>(varStepInitializer));
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
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::setParameters(const SimulationParameters& params, const std::string& name)
{
    getSimulation(name)->parameters = params;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::run()
{
    // fitnessGoal must be negative, the autotuning is searching for a Maximum.
    double fitnessGoal = -1.0;
    // default number of steps to simulate when normalizing test
    // duration or if normalization fails.
    unsigned defaultInitializerSteps = 5;

    prepareSimulations();
    if (!normalizeSteps(fitnessGoal, defaultInitializerSteps)) {
        LOG(Logger::WARN, "normalize Steps was not successful, default step number will be used");
        varStepInitializer->setMaxSteps(defaultInitializerSteps);
    }

    runTest();
    std::string best = getBestSim();
    runToCompletion(best);
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
std::string AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::getBestSim()
{
std::string bestSimulation;
double tmpFitness = -1 * DBL_MAX;
typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
for (IterType iter = simulations.begin(); iter != simulations.end(); iter++) {
    if (iter->second->fitness > tmpFitness) {
        tmpFitness = iter->second->fitness;
        bestSimulation = iter->first;
    }
}
LOG(Logger::DBG, "bestSimulation: " << bestSimulation)
return bestSimulation;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::runToCompletion(const std::string& optimizerName)
{
    // fixme: rework this
    boost::shared_ptr<Initializer<CELL_TYPE> > originalInitializer = varStepInitializer->getInitializer();
    varStepInitializer->setMaxSteps(originalInitializer->maxSteps());
    (*getSimulation(optimizerName)->simulationFactory)(simulations[optimizerName]->parameters);
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
unsigned AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::normalizeSteps(double goal, unsigned startStepNum)
{
    LOG(Logger::INFO, "normalizeSteps")
    if (startStepNum == 0) {
        throw std::invalid_argument("startSteps needs to be grater than zero");
    }

    SimulationPtr simulation = getSimulation("SerialSimulation");
    SimFactoryPtr factory = simulation->simulationFactory;
    unsigned steps = startStepNum;
    unsigned oldSteps = startStepNum;
    varStepInitializer->setMaxSteps(1);
    double variance = (*simulation->simulationFactory)(factory->parameters());
    double fitness = DBL_MAX;
    do {
        varStepInitializer->setMaxSteps(steps);
        fitness = (*factory)(simulation->parameters);
        oldSteps = steps;
        LOG(Logger::DBG, "steps: " << steps)
        steps = ((double) steps / fitness) * (double)goal;
        if (steps < 1) {
            steps =1;
        }
        LOG(Logger::DBG, "new calculated steps: " << steps);
        LOG(Logger::DBG, "fitness: " << fitness << " goal: " << goal);
        LOG(Logger::DBG, "variance: " << variance);
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

}

#endif

#endif
