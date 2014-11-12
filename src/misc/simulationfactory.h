#ifndef LIBGEODECOMP_MISC_SIMULATIONFACTORY_H
#define LIBGEODECOMP_MISC_SIMULATIONFACTORY_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>
#include <libgeodecomp/parallelization/cudasimulator.h>
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
    SimulationFactory()
    {
        std::vector<std::string> simulatorTypes;
        simulatorTypes << "SerialSimulator"
                       << "CacheBlockingSimulator"
                       << "CudaSimulator";
        parameterSet.addParameter("Simulator", simulatorTypes);
        parameterSet.addParameter("WavefrontWidth",  10, 1000);
        parameterSet.addParameter("WavefrontHeight", 10, 1000);
        parameterSet.addParameter("PipelineLength",   1,   30);
        parameterSet.addParameter("BlockDimX",        1,  128);
        parameterSet.addParameter("BlockDimX",        1,    8);
        parameterSet.addParameter("BlockDimX",        1,    8);
    }

    void addWriter(const ParallelWriter<CELL>& writer)
    {
        parallelWriters.push_back(boost::shared_ptr<ParallelWriter<CELL> >(writer.clone));
    }

    void addWriter(const Writer<CELL>& writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL> >(writer.clone));
    }

    /**
     * Returns a new simulator according to the previously specified
     * parameters. The user is expected to delete the simulator.
     */
    Simulator<CELL> *operator()(Initializer<CELL> *initializer)
    {
        Simulator<CELL> *sim = buildSimulator(initializer, parameterSet);
        // FIXME: add writers here?
        return sim;
    }

    double operator()(const SimulationParameters& params)
    {
        return 0;
    }

    SimulationParameters& parameters()
    {
        return parameterSet;
    }

private:
    SimulationParameters parameterSet;
    std::vector<boost::shared_ptr<ParallelWriter<CELL> > > parallelWriters;
    std::vector<boost::shared_ptr<Writer<CELL> > > writers;

    Simulator<CELL> *buildSimulator(
        Initializer<CELL> *initializer,
        const SimulationParameters& params) const
    {
        if (params["Simulator"] == "SerialSimulator") {
            return new SerialSimulator<CELL>(initializer);
        }

        if (params["Simulator"] == "CacheBlockingSimulator") {
            int pipelineLength  = params["PipelineLength"];
            int wavefrontWidth  = params["WavefrontWidth"];
            int wavefrontHeight = params["WavefrontHeight"];
            Coord<2> wavefrontDim(wavefrontWidth, wavefrontHeight);
            return new CacheBlockingSimulator<CELL>(initializer, pipelineLength, wavefrontDim);
        }

        if (params["Simulator"] == "CudaSimulator") {
            Coord<3> blockSize(params["BlockDimX"], params["BlockDimY"], params["BlockDimZ"]);
            return new CudaSimulator<CELL>(initializer, blockSize);
        }

        throw std::invalid_argument("unknown Simulator type");

    }
};

}

#endif
