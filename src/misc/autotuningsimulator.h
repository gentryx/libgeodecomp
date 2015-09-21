// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/logger.h>
#include <cfloat>

#define LIBGEODECOMP_DEBUG_LEVEL 4

namespace LibGeoDecomp{

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
class AutoTuningSimulator 
{
public:
    struct Result{
        Result(std::string name = std::string(),
            SimulationParameters param=SimulationParameters(), 
            double fit = DBL_MAX):
            nameOfSimulation(name),  
            parameters(param),
            fitness(fit)
        {}
        std::string nameOfSimulation;
        SimulationParameters parameters;
        double fitness;
    };
    
    typedef boost::shared_ptr<SimulationFactory<CELL_TYPE> > SimFactoryPtr;
    typedef boost::shared_ptr<Result> ResultPtr;
 
    struct Simulation {
        ResultPtr result;
        SimFactoryPtr simulationFactory;
    };
    typedef typename std::map<const std::string, Simulation>::iterator IterType;

    template<typename INITIALIZER>
    AutoTuningSimulator(INITIALIZER initializer):
        simulationSteps(10)
    {
        // FIXME simulation faktories erzeugen, wie soll herausgefunden werden welche verfügbar sind oder welche verwendet werden sollen...


        SimFactoryPtr ss_p(new SerialSimulationFactory<CELL_TYPE>(initializer));
        ResultPtr sr_p(new Result("SerialSimulation", ss_p->parameters()));
        Simulation ss_simulation;
        ss_simulation.result = sr_p;
        ss_simulation.simulationFactory = ss_p;
        simulations["SerialSimulation"] = ss_simulation;

        SimFactoryPtr cbs_p(new CacheBlockingSimulationFactory<CELL_TYPE>(initializer));
        ResultPtr cbs_result(new Result("CacheBlockingSimulation", cbs_p->parameters()));
        Simulation cbs_simulation;
        cbs_simulation.result = cbs_result;
        cbs_simulation.simulationFactory = cbs_p;
        simulations["CacheBlockingSimulation"] = cbs_simulation;
#ifdef WITH_CUDA
        SimFactoryPtr cudas_p(new CudaSimulationFactory<CELL_TYPE>(initializer));
        ResultPtr cudar_p(new Result("CudaSimulation", cudas_p->parameters()));
        Simulation cudas_simulation;
        simulations["CudaSimulation"].result = cudar_p;
        simulations["CudaSimulation"].simulationFactory = cudas_p;
#endif
    }

    ~AutoTuningSimulator()
    {}
    
    template<typename INITIALIZER>
    void addNewSimulation(Result& newResult, std::string name, INITIALIZER initializer)
    {
        // FIXME in newResult maybe no or wrong params!?
        std::string simName = newResult.nameOfSimulation;
        if (simName == "SerialSimulation")
        {
            SimFactoryPtr ss_p(new SerialSimulationFactory<CELL_TYPE>(initializer));
            ResultPtr cbr_p(new Result(newResult));
            simulations[name].result = cbr_p;
            simulations[name].simulationFactory = ss_p;
            return;
        }
        if (simName == "CacheBlockingSimulation")
        {
            SimFactoryPtr cbs_p(new CacheBlockingSimulationFactory<CELL_TYPE>(initializer));
            ResultPtr cbr_p(new Result(newResult));
            simulations[name].result = cbr_p;
            simulations[name].simulationFactory = cbs_p;
            return;
        }
#ifdef WITH_CUDA
        if (simName == "CudaSimulation")
        {
            SimFactoryPtr cudas_p(new CudaSimulationFactory<CELL_TYPE>(initializer));
            ResultPtr cbr_p(new Result(newResult));
            simulations[name].result = cbr_p;
            simulations[name].simulationFactory = cudas_p;
        }
#endif
        throw std::invalid_argument("unknown Simulator type");
    }

    void deleteAllSimulations()
    {
        simulations.clear();
    }

    SimulationParameters getSimulationParameters(const std::string simulationName)const
    {
        if (isInMap(simulationName))
            return simulations[simulationName].result->parameters;
        else
            return NULL;
    }
    
    void setSimulationSteps(unsigned steps)
    {
        simulationSteps = steps;
    }
    

    void setParameters(SimulationParameters params, std::string name)
    {
           
        if (isInMap(name))
            simulations[name].result->parameters = params;
        else
            throw std::invalid_argument(
                "AutotuningSimulatro<...>::setParameters(params,name) get invalid name");
    }

    void setOptimizer(OPTIMIZER_TYPE optimizer)
    {/*FIXME*/}

    virtual void run()
    {
        for (IterType iter = simulations.begin(); iter != simulations.end(); iter++)
        {
            LOG(Logger::DBG, iter->first)
            OPTIMIZER_TYPE optimizer(iter->second.result->parameters);
            iter->second.result->parameters = optimizer(
                simulationSteps, 
                *iter->second.simulationFactory);
            iter->second.result->fitness = optimizer.getFitness();
            LOG(Logger::DBG, "Result of the "<< iter->second.result->nameOfSimulation 
                << ": " << iter->second.result->fitness << std::endl 
                << "new Parameters:"<< std::endl << iter->second.result->parameters 
                << std::endl)
        }
    }

private:
    std::map<const std::string, Simulation> simulations;
    std::vector<ResultPtr> results;
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
