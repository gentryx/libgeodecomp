#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H

#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>
// There are problems if cuda is not installed on the system
// FIXME it ned to be checked by the preprocessor
//#include <libgeodecomp/parallelization/cudasimulator.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

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
    {
        std::vector<std::string> simulatorTypes;
        simulatorTypes << "SerialSimulator"
//                       << "CudaSimulator"
                       << "CacheBlockingSimulator";
        parameterSet.addParameter("Simulator", simulatorTypes);
        parameterSet.addParameter("WavefrontWidth",  10, 1000);
        parameterSet.addParameter("WavefrontHeight", 10, 1000);
        parameterSet.addParameter("PipelineLength",   1,   30);
//        parameterSet.addParameter("BlockDimX",        1,  128);
//        parameterSet.addParameter("BlockDimY",        1,    8);
//        parameterSet.addParameter("BlockDimz",        1,    8);
    }

    ~SimulationFactory()
    {
        // fixme: we can't delete the initializer here because of the missing clone() in initializer
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
        Simulator<CELL> *sim = buildSimulator(initializer->clone(), parameterSet);
        return sim;
    }

    double operator()(const SimulationParameters& params)
    {
        Simulator<CELL> *sim = buildSimulator(initializer->clone(), params);
        Chronometer chrono;

        {
            TimeCompute t(&chrono);
            sim->run();
        }

        delete sim;
        return chrono.interval<TimeCompute>();
    }

    SimulationParameters& parameters()
    {
        return parameterSet;
    }

private:
    ClonableInitializer<CELL> *initializer;
    SimulationParameters parameterSet;
    std::vector<boost::shared_ptr<ParallelWriter<CELL> > > parallelWriters;
    std::vector<boost::shared_ptr<Writer<CELL> > > writers;
    std::vector<boost::shared_ptr<Steerer<CELL> > > steerers;

    Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
       if (params["Simulator"] == "SerialSimulator") {
            SerialSimulator<CELL> *sSim = new SerialSimulator<CELL>(initializer);
            for (unsigned i = 0; i < writers.size(); ++i)
                sSim->addWriter(writers[i].get()->clone());
            for (unsigned i = 0; i < steerers.size(); ++i)
                sSim->addSteerer(steerers[i].get()->clone());
            return sSim;
        }

        if (params["Simulator"] == "CacheBlockingSimulator") {
            int pipelineLength  = params["PipelineLength"];
            int wavefrontWidth  = params["WavefrontWidth"];
            int wavefrontHeight = params["WavefrontHeight"];
            Coord<2> wavefrontDim(wavefrontWidth, wavefrontHeight);
            CacheBlockingSimulator<CELL> *cbSim = 
                new CacheBlockingSimulator<CELL>(
                    initializer, 
                    pipelineLength, 
                    wavefrontDim);
            for(unsigned i = 0; i < writers.size(); ++i){
                cbSim->addWriter(writers[i].get()->clone());
            for (unsigned i = 0; i < steerers.size(); ++i)
                cbSim->addSteerer(steerers[i].get()->clone());
            }
            return cbSim;
        }

        //if (params["Simulator"] == "CudaSimulator") {
        //    Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
        //    CudaSimulator<CELL> * cSim = new CudaSimulator<CELL>(initializer, blockSize);
        //    for (unsigned i = 0; i < writers.size(); ++i)
        //        cSim->addWriter(writers[i].get()->clone());
        //    for (unsigned i = 0; i < steerers.size(); ++i)
        //        cSim->addSteerer(steerers[i].get()->clone());
        //    return cSim;
        //}

        // FIXME: Something need to be done with the parallelWriters!

        throw std::invalid_argument("unknown Simulator type");
    }
};

}

#endif
