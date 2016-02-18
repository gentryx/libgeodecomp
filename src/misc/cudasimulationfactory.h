#ifndef LIBGEODECOMP_MISC_CUDASIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_CUDASIMULATIONFACTORY_H

#include <libgeodecomp/misc/simulationfactory.h>

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA

#include <libgeodecomp/parallelization/cudasimulator.h>

namespace LibGeoDecomp {

template<typename CELL>
class CudaSimulationFactory : public SimulationFactory<CELL>
{
public:
    using SimulationFactory<CELL>::addSteerers;
    using SimulationFactory<CELL>::addWriters;

    CudaSimulationFactory<CELL>(boost::shared_ptr<ClonableInitializer<CELL> > initializer) :
        SimulationFactory<CELL>(initializer)
    {
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimX", 1, 128);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimY", 1,   8);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimZ", 1,   8);
    }

    virtual double operator()(const SimulationParameters& params)
    {
        LOG(Logger::DBG, "SimulationFactory::operator(params)");
        boost::shared_ptr<ClonableInitializer<CELL> > init(SimulationFactory<CELL>::initializer->clone());
        Simulator<CELL> *sim = buildSimulator(init, params);
        LOG(Logger::DBG, "sim get buildSimulator(initializer->clone(), params)")
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            LOG(Logger::DBG,"next step is sim->run()")
            try {
                sim->run();
            } catch(const std::runtime_error& error){
                LOG(Logger::INFO,"runtime error detcted")
                delete sim;
                return DBL_MAX *-1.0;
            }
        }

        LOG(Logger::DBG,"now deleting sim")
        delete sim;
        return chrono.interval<TimeCompute>() * -1.0;
    }

protected:
    virtual Simulator<CELL> *buildSimulator(
        boost::shared_ptr<ClonableInitializer<CELL> > initializer,
        const SimulationParameters& params) const
    {
        Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
        CudaSimulator<CELL> *sim = new CudaSimulator<CELL>(initializer->clone(), blockSize);

        addWriters(sim);
        addSteerers(sim);

        return sim;
    }
};

}

#endif
#endif

#endif
