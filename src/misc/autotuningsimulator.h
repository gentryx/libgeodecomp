// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#define LIBGEODECOMP_MISC_AUTOTUNINGSIMULATOR_H
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/io/logger.h>
#include <iostream>
#include <sstream>

//#define LIBGEODECOMP_DEBUG_LEVEL 4

namespace LibGeoDecomp{

template<typename CELL_TYPE,typename OPTIMIZER_TYPE>
class AutoTuningSimulator 
{
public:
    template<typename INITIALIZER>
    AutoTuningSimulator(INITIALIZER initializer):
        simFab(SimulationFactory<CELL_TYPE>(initializer)),
        params(simFab.parameters()),
        optimizer(OPTIMIZER_TYPE(params))
    {
        params["Simulator"]="CacheBlockingSimulator";
    }

    SimulationParameters getSimulationParameters()
    {
        return params;
    }
    
    void setParameters(SimulationParameters params)
    {
        optimizer = OPTIMIZER_TYPE(params);
    }

    void setOptimizer(OPTIMIZER_TYPE optimizer)
    {/*FIXME*/}

    virtual void run()
    {
        params = optimizer(10, simFab);
        std::stringstream log;
        log << "Result of the Optimization: " << optimizer.getFitness()
            << std::endl
            << "new Parameters:"<< std::endl << params << std::endl;
        LOG(Logger::DBG, log.str())
    }

private:
    SimulationFactory<CELL_TYPE> simFab;
    SimulationParameters params;
    OPTIMIZER_TYPE optimizer;
}; //AutoTunigSimulator
} // namespace LibGeoDecomp

#endif
