#ifndef LIBGEODECOMP_MISC_CUDASIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_CUDASIMULATIONFACTORY_H

#include <libgeodecomp/misc/simulationfactory.h>

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA

#include <libgeodecomp/parallelization/cudasimulator.h>

namespace LibGeoDecomp {

/**
 * As its name implies this class helps with instantiating
 * CUDASimulators and lists all drivable/optimizable parameters.
 */
template<typename CELL>
class CUDASimulationFactory : public SimulationFactory<CELL>
{
public:
    using SimulationFactory<CELL>::addSteerers;
    using SimulationFactory<CELL>::addWriters;

    CUDASimulationFactory<CELL>(boost::shared_ptr<ClonableInitializer<CELL> > initializer) :
        SimulationFactory<CELL>(initializer)
    {
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimX", 1, 128);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimY", 1,   8);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimZ", 1,   8);
    }

    virtual double operator()(const SimulationParameters& params)
    {
        boost::shared_ptr<ClonableInitializer<CELL> > init(SimulationFactory<CELL>::initializer->clone());
        boost::shared_ptr<Simulator<CELL> > sim(buildSimulator(init, params));
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            try {
                sim->run();
            } catch(const std::runtime_error& error){
                return std::numeric_limits<double>::min();
            }
        }

        return chrono.interval<TimeCompute>() * -1.0;
    }

    std::string name() const
    {
        return "CUDASimulator";
    }

protected:
    virtual Simulator<CELL> *buildSimulator(
        boost::shared_ptr<ClonableInitializer<CELL> > initializer,
        const SimulationParameters& params) const
    {
        Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
        CUDASimulator<CELL> *sim = new CUDASimulator<CELL>(initializer->clone(), blockSize);

        addWriters(sim);
        addSteerers(sim);

        return sim;
    }
};

}

#endif
#endif

#endif
