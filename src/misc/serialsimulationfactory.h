#ifndef LIBGEODECOMP_MISC_SERIALSIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SERIALSIMULATIONFACTORY_H

#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

namespace LibGeoDecomp {

/**
 * This helper class will manufacture SerialSimulators, used by our
 * auto-tuning infrastructure.
 */
template<typename CELL>
class SerialSimulationFactory : public SimulationFactory<CELL>
{
public:
    using SimulationFactory<CELL>::addSteerers;
    using SimulationFactory<CELL>::addWriters;
    typedef typename SimulationFactory<CELL>::InitPtr InitPtr;

    explicit
    SerialSimulationFactory(InitPtr initializer) :
        SimulationFactory<CELL>(initializer)
    {
        // SerialSimulator has no parameters to optimize
    }

    std::string name() const
    {
        return "SerialSimulator";
    }

protected:
    virtual Simulator<CELL> *buildSimulator(
        InitPtr initializer,
        const SimulationParameters& params) const
    {
        SerialSimulator<CELL> *sim = new SerialSimulator<CELL>(initializer->clone());

        addWriters(sim);
        addSteerers(sim);

        return sim;
    }
};

}

#endif
