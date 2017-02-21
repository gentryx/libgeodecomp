#ifndef LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_AUTOTUNINGSIMULATOR_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/cacheblockingsimulationfactory.h>
#include <libgeodecomp/misc/cudasimulationfactory.h>
#include <libgeodecomp/misc/limits.h>
#include <libgeodecomp/misc/serialsimulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/io/logger.h>
#include <cfloat>

namespace LibGeoDecomp {

namespace AutoTuningSimulatorHelpers {

/**
 * A helper flass which encapsulates a simulation factory and the
 * associated fitness values.
 */
template<typename CELL_TYPE>
class Simulation{
public:
    typedef typename SharedPtr<SimulationFactory<CELL_TYPE> >::Type SimFactoryPtr;

    Simulation(
        const std::string& name,
        SimFactoryPtr simFactory,
        SimulationParameters param,
        double fit = Limits<double>::getMax()):
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
 *
 * fixme: shouldn't we inherit from Monolithic- or DistributedSimulator?
 */
template<typename CELL_TYPE, typename OPTIMIZER_TYPE>
class AutoTuningSimulator
{
public:
    friend class AutotuningSimulatorWithoutCUDATest;
    friend class AutotuningSimulatorWithCUDATest;

    typedef AutoTuningSimulatorHelpers::Simulation<CELL_TYPE>  Simulation;
    typedef typename SharedPtr<SimulationFactory<CELL_TYPE> >::Type SimFactoryPtr;
    typedef typename SharedPtr<Simulation>::Type SimulationPtr;

    typedef typename Simulator<CELL_TYPE>::InitPtr InitPtr;
    typedef typename Simulator<CELL_TYPE>::SteererPtr SteererPtr;

    explicit
    AutoTuningSimulator(Initializer<CELL_TYPE> *initializer, unsigned optimizationSteps = 10);

    void addWriter(ParallelWriter<CELL_TYPE> *writer);

    void addWriter(Writer<CELL_TYPE> *writer);

    void addSteerer(const Steerer<CELL_TYPE> *steerer);

    void run();

private:
    std::map<const std::string, SimulationPtr> simulations;
    unsigned optimizationSteps; // maximum number of Steps for the optimizer
    typename SharedPtr<VarStepInitializerProxy<CELL_TYPE> >::Type varStepInitializer;
    std::vector<typename SharedPtr<ParallelWriter<CELL_TYPE> >::Type> parallelWriters;
    std::vector<typename SharedPtr<Writer<CELL_TYPE> >::Type> writers;
    std::vector<typename SharedPtr<Steerer<CELL_TYPE> >::Type> steerers;

    template<typename FACTORY_TYPE>
    void addSimulation(const std::string& name, const FACTORY_TYPE& factory)
    {
        SimFactoryPtr simFactoryPtr(new FACTORY_TYPE(factory));
        SimulationPtr simulationPtr(new Simulation(
                                        name,
                                        simFactoryPtr,
                                        simFactoryPtr->parameters()));
        simulations[name] = simulationPtr;
    }

    template<typename FACTORY_TYPE>
    void addSimulation(const FACTORY_TYPE& factory)
    {
        addSimulation(factory.name(), factory);
    }

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
    addSimulation(SerialSimulationFactory<CELL_TYPE>(varStepInitializer));
#ifdef LIBGEODECOMP_WITH_THREADS
    addSimulation(CacheBlockingSimulationFactory<CELL_TYPE>(varStepInitializer));
#endif

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
    addSimulation(CUDASimulationFactory<CELL_TYPE>(varStepInitializer));
#endif
#endif
}


template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addWriter(ParallelWriter<CELL_TYPE> *writer)
{
    parallelWriters.push_back(typename SharedPtr<ParallelWriter<CELL_TYPE> >::Type(writer));
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addWriter(Writer<CELL_TYPE> *writer)
{
    writers.push_back(typename SharedPtr<Writer<CELL_TYPE> >::Type(writer));
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::addSteerer(const Steerer<CELL_TYPE> *steerer)
{
    steerers.push_back(SteererPtr(steerer));
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
    double tmpFitness = Limits<double>::getMin();
    typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;

    for (IterType iter = simulations.begin(); iter != simulations.end(); iter++) {
        if (iter->second->fitness > tmpFitness) {
            tmpFitness = iter->second->fitness;
            bestSimulation = iter->first;
        }
    }

    LOG(Logger::DBG, "bestSimulation: " << bestSimulation);
    return bestSimulation;
}

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
void AutoTuningSimulator<CELL_TYPE, OPTIMIZER_TYPE>::runToCompletion(const std::string& optimizerName)
{
    varStepInitializer->resetMaxSteps();
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
    double fitness = Limits<double>::getMax();

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
