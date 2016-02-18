// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H

#include <libgeodecomp/io/clonableinitializer.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>
#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
#include <libgeodecomp/parallelization/cudasimulator.h>
#endif
#endif
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/logger.h>

namespace LibGeoDecomp {

/**
 * A convenience class for setting up all objects (e.g. writers and
 * steerers) necessary for conduction a simulation.
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
            // fixme: we should clone here
            simulator->addSteerer(&**i);
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

/**
 * This helper class will manufacture SerialSimulators, used by our
 * auto-tuning infrastructure.
 *
 * fixme: move factories to dedicated files
 */
template<typename CELL>
class SerialSimulationFactory : public SimulationFactory<CELL>
{
public:
    using SimulationFactory<CELL>::addSteerers;
    using SimulationFactory<CELL>::addWriters;

    SerialSimulationFactory(boost::shared_ptr<ClonableInitializer<CELL> > initializer) :
        SimulationFactory<CELL>(initializer)
    {
        // Serial Simulation has no Parameters to optimize
    }

    virtual ~SerialSimulationFactory()
    {}

protected:

    virtual Simulator<CELL> *buildSimulator(
        boost::shared_ptr<ClonableInitializer<CELL> > initializer,
        const SimulationParameters& params) const
    {
        SerialSimulator<CELL> *sim = new SerialSimulator<CELL>(initializer->clone());

        addWriters(sim);
        addSteerers(sim);

        return sim;
    }
};

#ifdef LIBGEODECOMP_WITH_THREADS
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
        SimulationFactory<CELL>::parameterSet.addParameter("PipelineLength",  1, 30);
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
#endif

#ifdef __CUDACC__
#ifdef LIBGEODECOMP_WITH_CUDA
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

#endif
#endif

}

#endif
