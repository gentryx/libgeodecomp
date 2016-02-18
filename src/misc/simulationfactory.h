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

    typedef std::vector<boost::shared_ptr<ParallelWriter<CELL> > > ParallelWritersVec;
    typedef std::vector<boost::shared_ptr<Writer<CELL> > > WritersVec;
    typedef std::vector<boost::shared_ptr<Steerer<CELL> > > SteerersVec;

    SimulationFactory(boost::shared_ptr<ClonableInitializer<CELL> > initializer) :
        initializer(initializer)
    {}

    virtual ~SimulationFactory()
    {}

    void addWriter(const ParallelWriter<CELL>& writer)
    {
        parallelWriters.push_back(boost::shared_ptr<ParallelWriter<CELL> >(writer.clone()));
    }

    void addWriter(const Writer<CELL>& writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL> >(writer.clone()));
    }

    void addSteerer(const Steerer<CELL>& steerer)
    {
        steerers.push_back(boost::shared_ptr<Steerer<CELL> >(steerer.clone()));
    }

    /**
     * Returns a new simulator according to the previously specified
     * parameters. The user is expected to delete the simulator.
     */
    Simulator<CELL> *operator()()
    {
        LOG(Logger::DBG, "SimulationFactory::operator()")
        Simulator<CELL> *sim = buildSimulator(initializer, parameterSet);
        return sim;
    }

    // fixme: move this functionality to another class, possibly inside AutoTuningSimulator?
    virtual double operator()(const SimulationParameters& params)
    {
        LOG(Logger::DBG, "SimulationFactory::operator(params)");
        Simulator<CELL> *sim = buildSimulator(initializer, params);
        LOG(Logger::DBG, "sim get buildSimulator(initializer->clone(), params)")
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            LOG(Logger::DBG,"next step is sim->run()")
            sim->run();
        }

        LOG(Logger::DBG,"now deleting sim")
        delete sim;
        return chrono.interval<TimeCompute>() * -1.0;
    }

    const SimulationParameters& parameters() const
    {
        return parameterSet;
    }

protected:
    boost::shared_ptr<ClonableInitializer<CELL> > initializer;
    SimulationParameters parameterSet;
    ParallelWritersVec parallelWriters;
    WritersVec writers;
    SteerersVec steerers;

    virtual Simulator<CELL> *buildSimulator(
        boost::shared_ptr<ClonableInitializer<CELL> > initializer,
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
