#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H


//#define WITH_CUDA

#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>
// There are problems if cuda is not installed on the system
// FIXME it ned to be checked by the preprocessor
#ifdef WITH_CUDA
#include <libgeodecomp/parallelization/cudasimulator.h>
#endif
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/logger.h>


#define LIBGEODECOMP_DEBUG_LEVEL 4
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

    ~SimulationFactory()
    {
        // FIXME: we can't delete the initializer here because of the missing clone() in initializer
        // delete initializer;
    }

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
        Simulator<CELL> *sim = buildSimulator(initializer->clone(), parameterSet);
        return sim;
    }

    double operator()(const SimulationParameters& params)
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

        LOG(Logger::DBG,"now delet sim")
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
protected:
    
    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
        SerialSimulator<CELL> *sSim = new SerialSimulator<CELL>(initializer);
        for (unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i)
            sSim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
        for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
            sSim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
        return sSim;
    }
};

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
protected:
    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
        int pipelineLength  = params["PipelineLength"];
        int wavefrontWidth  = params["WavefrontWidth"];
        int wavefrontHeight = params["WavefrontHeight"];
        Coord<2> wavefrontDim(wavefrontWidth, wavefrontHeight);
        CacheBlockingSimulator<CELL> *cbSim = 
            new CacheBlockingSimulator<CELL>(
                initializer, 
                pipelineLength, 
                wavefrontDim);
        for(unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i){
            cbSim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
        for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
            cbSim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
        }
        return cbSim;
    }
};

// FIXME: everything in this file which is depends on CUDA is not tested!
#ifdef WITH_CUDA
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
protected:
    virtual Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
            LOG(Logger::DBG, "enter CudaSimulationFactory::build()")
            Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
            LOG(Logger::DBG, "generate new CudaSimulator")
            CudaSimulator<CELL> * cSim = new CudaSimulator<CELL>(initializer, blockSize);
            LOG(Logger::DBG, "addWriters")
            for (unsigned i = 0; i < SimulationFactory<CELL>::writers.size(); ++i)
                cSim->addWriter(SimulationFactory<CELL>::writers[i].get()->clone());
            LOG(Logger::DBG, "addSteers")
            for (unsigned i = 0; i < SimulationFactory<CELL>::steerers.size(); ++i)
                cSim->addSteerer(SimulationFactory<CELL>::steerers[i].get()->clone());
            LOG(Logger::DBG, "return cSim")
            return cSim;

    }
};
#endif
}//namespace LibGeoDecomp

#endif
