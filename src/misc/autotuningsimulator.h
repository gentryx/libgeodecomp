// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/logger.h>
#include <cfloat>

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

    template<typename INITIALIZER>
    AutoTuningSimulator(INITIALIZER initializer):
        simulationSteps(10)
    {
        addNewSimulation("SerialSimulation",
            "SerialSimulation",
            initializer);

        addNewSimulation("CacheBlockingSimulation",
            "CacheBlockingSimulation",
            initializer);

#ifdef LIBGEODECOMP_WITH_CUDA
        addNewSimulation("CudaSimulationFactory",
            "CudaSimulationFactory",
            initializer);
#endif
    }

    ~AutoTuningSimulator()
    {}
    
    template<typename INITIALIZER>
    void addNewSimulation(std::string name, std::string typeOfSimulation, INITIALIZER initializer)
    {
        if (typeOfSimulation == "SerialSimulation")
        {
            SimFactoryPtr simFac_p(new SerialSimulationFactory<CELL_TYPE>(initializer));
            SimulationPtr sim_p(new Simulation(
                    typeOfSimulation,
                    simFac_p,
                    simFac_p->parameters()));
            simulations[name] = sim_p;
            return;
        }
        
        if (typeOfSimulation == "CacheBlockingSimulation")
        {
            SimFactoryPtr simFac_p(new CacheBlockingSimulationFactory<CELL_TYPE>(initializer));
            SimulationPtr sim_p(new Simulation(
                    typeOfSimulation,
                    simFac_p,
                    simFac_p->parameters()));
            simulations[name] = sim_p;
            return;
        }

#ifdef LIBGEODECOMP_WITH_CUDA
         if (typeOfSimulation == "CudaSimulation")
         {
            SimFactoryPtr simFac_p(new CudaSimulationFactory<CELL_TYPE>(initializer));
            SimulationPtr sim_p(new Simulation(
                    typeOfSimulation,
                    simFac_p,
                    simFac_p->parameters()));
            simulations[name] = sim_p;
            return;
         }
#endif

        throw std::invalid_argument("SimulationFactory::addNewSimulation(): unknown simulator type");
    }

    void deleteAllSimulations()
    {
        simulations.clear();
    }

    SimulationParameters getSimulationParameters(std::string simulationName)
    {
        if (isInMap(simulationName))
            return simulations[simulationName]->parameters;
        else
            throw std::invalid_argument("getSimulationParameters(simulationName) get invalid simulationName");
    }
    
    void setSimulationSteps(unsigned steps)
    {
        simulationSteps = steps;
    }
    

    void setParameters(SimulationParameters params, std::string name)
    {
           
        if (isInMap(name))
            simulations[name]->parameters = params;
        else
            throw std::invalid_argument(
                "AutotuningSimulatro<...>::setParameters(params,name) get invalid name");
    }

    virtual void run()
    {
        typedef typename std::map<const std::string, SimulationPtr>::iterator IterType;
        for (IterType iter = simulations.begin(); iter != simulations.end(); iter++)
        {
            LOG(Logger::DBG, iter->first)
            
            OPTIMIZER_TYPE optimizer(iter->second->parameters);
            iter->second->parameters = optimizer(
                simulationSteps, 
                *iter->second->simulationFactory);
            iter->second->fitness = optimizer.getFitness();
            
            LOG(Logger::DBG, "Result of the " << iter->second->simulationType 
                << ": " << iter->second->fitness << std::endl 
                << "new Parameters:"<< std::endl << iter->second->parameters 
                << std::endl)
        }
    }

private:
    std::map<const std::string, SimulationPtr> simulations;
    unsigned simulationSteps;

    bool isInMap(const std::string name)const
    {
        if (simulations.find(name) == simulations.end())
            return false;
        else
            return true;
    }
}; //AutoTunigSimulator
} // namespace LibGeoDecomp

#endif
