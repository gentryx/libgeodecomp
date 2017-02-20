#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H

#include <libgeodecomp/io/clonableinitializer.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>

namespace LibGeoDecomp {

/**
 * A SimulationFactory sets up all objects (e.g. Writers and
 * Steerers) necessary for conducting a simulation.
 */
template<typename CELL>
class SimulationFactory : public Optimizer::Evaluator
{
public:
    friend class SimulationFactoryWithoutCudaTest;
    friend class SimulationFactoryWithCudaTest;

    typedef typename SharedPtr<ClonableInitializer<CELL> >::Type InitPtr;
    typedef std::vector<typename SharedPtr<ParallelWriter<CELL> >::Type> ParallelWritersVec;
    typedef std::vector<typename SharedPtr<Writer<CELL> >::Type> WritersVec;
    typedef std::vector<typename SharedPtr<Steerer<CELL> >::Type> SteerersVec;

    explicit
    SimulationFactory(InitPtr initializer) :
        initializer(initializer)
    {}

    virtual ~SimulationFactory()
    {}

    void addWriter(const ParallelWriter<CELL>& writer)
    {
        parallelWriters.push_back(typename SharedPtr<ParallelWriter<CELL> >::Type(writer.clone()));
    }

    void addWriter(const Writer<CELL>& writer)
    {
        writers.push_back(typename SharedPtr<Writer<CELL> >::Type(writer.clone()));
    }

    void addSteerer(const Steerer<CELL>& steerer)
    {
        steerers.push_back(typename SharedPtr<Steerer<CELL> >::Type(steerer.clone()));
    }

    /**
     * Returns a new simulator according to the previously specified
     * parameters. The user is expected to delete the simulator.
     */
    Simulator<CELL> *operator()()
    {
        Simulator<CELL> *sim = buildSimulator(initializer, parameterSet);
        return sim;
    }

    virtual double operator()(const SimulationParameters& params)
    {
        typename SharedPtr<Simulator<CELL> >::Type sim(buildSimulator(initializer, params));
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            sim->run();
        }

        return chrono.interval<TimeCompute>() * -1.0;
    }

    const SimulationParameters& parameters() const
    {
        return parameterSet;
    }

protected:
    InitPtr initializer;
    SimulationParameters parameterSet;
    ParallelWritersVec parallelWriters;
    WritersVec writers;
    SteerersVec steerers;

    virtual Simulator<CELL> *buildSimulator(
        InitPtr initializer,
        const SimulationParameters& params) const = 0;

    void addSteerers(MonolithicSimulator<CELL> *simulator) const
    {
        for (typename SteerersVec::const_iterator i = steerers.begin(); i != steerers.end(); ++i) {
            simulator->addSteerer((*i)->clone());
        }
    }

    void addWriters(MonolithicSimulator<CELL> *simulator) const
    {
        for (typename WritersVec::const_iterator i = writers.begin(); i != writers.end(); ++i) {
            // fixme: we should clone here
            simulator->addWriter(&**i);
        }
    }
    void addWriters(DistributedSimulator<CELL> *simulator) const
    {
        for (typename ParallelWritersVec::const_iterator i = parallelWriters.begin(); i != parallelWriters.end(); ++i) {
            // fixme: we should clone here
            simulator->addWriter(&**i);
        }
    }
};

}

#endif
