// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H

#include <libgeodecomp/io/clonableinitializerwrapper.h>
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
    template<typename INITIALIZER>
    SimulationFactory(INITIALIZER initializer) :
        initializer(ClonableInitializerWrapper<INITIALIZER>::wrap(initializer))
    {}

    virtual ~SimulationFactory()
    {
        delete initializer;
    }

    void addWriter(const ParallelWriter<CELL>& writer)
    {
        parallelWriters.push_back(boost::shared_ptr<ParallelWriter<CELL> >(writer.clone()));
    }

    void addWriter(const Writer<CELL>& writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL> >(writer.clone()));
    }

    void addSteerer(Steerer<CELL>& steerer) //FIXME why is const on steerer not working?
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
        Simulator<CELL> *sim = buildSimulator(initializer->clone(), parameterSet);
        return sim;
    }

    virtual double operator()(const SimulationParameters& params)
    {
        LOG(Logger::DBG, "SimulationFactory::operator(params)")
        Simulator<CELL> *sim = buildSimulator(initializer->clone(), params);
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

    SimulationParameters& parameters()
    {
        return parameterSet;
    }
protected:
    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const = 0;

    ClonableInitializer<CELL> *initializer;
    SimulationParameters parameterSet;
    std::vector<boost::shared_ptr<ParallelWriter<CELL> > > parallelWriters;
    std::vector<boost::shared_ptr<Writer<CELL> > > writers;
    // FIXME: Something need to be done with the parallelWriters in subclasses!
    std::vector<boost::shared_ptr<Steerer<CELL> > > steerers;
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

    template<typename INITIALIZER>
    SerialSimulationFactory(INITIALIZER initializer):
        SimulationFactory<CELL>(initializer)
    {
        // Serial Simulation has no Parameters to optimize
    }

    virtual ~SerialSimulationFactory(){}
protected:

    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
        SerialSimulator<CELL> *sim = new SerialSimulator<CELL>(initializer);
        for (unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i)
            sim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
        for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
            sim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
        return sim;
    }
};

#ifdef LIBGEODECOMP_WITH_THREADS
template<typename CELL>
class CacheBlockingSimulationFactory : public SimulationFactory<CELL>
{
public:
    template<typename INITIALIZER>
    CacheBlockingSimulationFactory<CELL>(INITIALIZER initializer):
        SimulationFactory<CELL>(initializer)
    {
        SimulationFactory<CELL>::parameterSet.addParameter("WavefrontWidth", 10, 1000);
        SimulationFactory<CELL>::parameterSet.addParameter("WavefrontHeight",10, 1000);
        SimulationFactory<CELL>::parameterSet.addParameter("PipelineLength",  1, 30);
    }
    virtual ~CacheBlockingSimulationFactory(){}
protected:
    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
        int pipelineLength  = params["PipelineLength"];
        int wavefrontWidth  = params["WavefrontWidth"];
        int wavefrontHeight = params["WavefrontHeight"];
        Coord<2> wavefrontDim(wavefrontWidth, wavefrontHeight);
        CacheBlockingSimulator<CELL> *sim =
            new CacheBlockingSimulator<CELL>(
                initializer,
                pipelineLength,
                wavefrontDim);
        for(unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i){
            sim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
        for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
            sim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
        }
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
    template<typename INITIALIZER>
    CudaSimulationFactory<CELL>(INITIALIZER initializer):
        SimulationFactory<CELL>(initializer)
    {
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimX", 1, 128);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimY", 1,   8);
        SimulationFactory<CELL>::parameterSet.addParameter("BlockDimZ", 1,   8);
    }

    virtual ~CudaSimulationFactory(){}

    virtual double operator()(const SimulationParameters& params)
    {
        LOG(Logger::DBG, "SimulationFactory::operator(params)")
        Simulator<CELL> *sim = buildSimulator(SimulationFactory<CELL>::initializer->clone(), params);
        LOG(Logger::DBG, "sim get buildSimulator(initializer->clone(), params)")
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            LOG(Logger::DBG,"next step is sim->run()")
            try{
                sim->run();
            }
            catch(const std::runtime_error& error){
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
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
            LOG(Logger::DBG, "enter CudaSimulationFactory::build()")
            Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
            LOG(Logger::DBG, "generate new CudaSimulator")
            CudaSimulator<CELL> * sim = new CudaSimulator<CELL>(initializer, blockSize);
            LOG(Logger::DBG, "addWriters")
            for (unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i)
                sim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
            LOG(Logger::DBG, "addSteers")
            for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
                sim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
            LOG(Logger::DBG, "return Simulator from CudaSimulationFactory::buildSimulator()")
            return sim;

    }
};
#endif // LIBGEODECOMP_WITH_CUDA
#endif // __CUDACC__
}//namespace LibGeoDecomp

#endif
