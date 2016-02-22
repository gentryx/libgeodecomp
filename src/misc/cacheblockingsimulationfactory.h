#ifndef LIBGEODECOMP_MISC_CACHEBLOCKINGSIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_CACHEBLOCKINGSIMULATIONFACTORY_H

#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>

#ifdef LIBGEODECOMP_WITH_THREADS

namespace LibGeoDecomp {

/**
 * Customized factory for instantiating a CacheBlockingSimulator
 */
template<typename CELL>
class CacheBlockingSimulationFactory : public SimulationFactory<CELL>
{
public:
    using SimulationFactory<CELL>::addSteerers;
    using SimulationFactory<CELL>::addWriters;

    CacheBlockingSimulationFactory<CELL>(boost::shared_ptr<ClonableInitializer<CELL> > initializer):
        SimulationFactory<CELL>(initializer)
    {
        SimulationFactory<CELL>::parameterSet.addParameter("WavefrontWidth", 10, 1000);
        SimulationFactory<CELL>::parameterSet.addParameter("WavefrontHeight",10, 1000);
        SimulationFactory<CELL>::parameterSet.addParameter("PipelineLength",  1,   30);
    }

    std::string name() const
    {
        return "CacheBlockingSimulator";
    }

protected:
    virtual Simulator<CELL> *buildSimulator(
        boost::shared_ptr<ClonableInitializer<CELL> > initializer,
        const SimulationParameters& params) const
    {
        int pipelineLength  = params["PipelineLength"];
        int wavefrontWidth  = params["WavefrontWidth"];
        int wavefrontHeight = params["WavefrontHeight"];

        Coord<2> wavefrontDim(wavefrontWidth, wavefrontHeight);
        CacheBlockingSimulator<CELL> *sim =
            new CacheBlockingSimulator<CELL>(
                initializer->clone(),
                pipelineLength,
                wavefrontDim);

        addWriters(sim);
        addSteerers(sim);

        return sim;
    }
};

}

#endif

#endif
