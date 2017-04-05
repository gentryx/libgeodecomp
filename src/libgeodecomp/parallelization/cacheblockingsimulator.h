#ifndef LIBGEODECOMP_PARALLELIZATION_CACHEBLOCKINGSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_CACHEBLOCKINGSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_THREADS

#include <omp.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * CacheBlockingSimulator is an experimental simulator to explore the
 * infrastructure required to implement a pipelined wavefront update
 * algorith and which benefits is may provide.
 */
template<typename CELL>
class CacheBlockingSimulator : public MonolithicSimulator<CELL>
{
public:
    friend class CacheBlockingSimulatorTest;

    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    typedef typename TopologiesHelpers::Topology<3, Topology::template WrapsAxis<0>::VALUE, Topology::template WrapsAxis<1>::VALUE, true> BufferTopology;
    typedef Grid<CELL, Topology> GridType;
    typedef DisplacedGrid<CELL, BufferTopology> BufferType;
    typedef std::vector<std::vector<Region<3> > > WavefrontFrames;
    static const int DIM = Topology::DIM;

    using MonolithicSimulator<CELL>::NANO_STEPS;
    using MonolithicSimulator<CELL>::chronometer;

    CacheBlockingSimulator(
        Initializer<CELL> *initializer,
        unsigned pipelineLength,
        const Coord<DIM - 1>& wavefrontDim) :
        MonolithicSimulator<CELL>(initializer),
        buffers(static_cast<std::size_t>(omp_get_max_threads())),
        pipelineLength(pipelineLength),
        wavefrontDim(wavefrontDim)
    {
        Coord<DIM> dim = initializer->gridBox().dimensions;
        curGrid = new GridType(dim);
        newGrid = new GridType(dim);
        initializer->grid(curGrid);
        initializer->grid(newGrid);

        Coord<DIM> bufferDim;

        for (int i = 0; i < DIM - 1; ++i) {
            bufferDim[i] = wavefrontDim[i] + 2 * static_cast<int>(pipelineLength) - 2;
        }
        bufferDim[DIM - 1] = static_cast<int>(pipelineLength) * 4 - 4;

        for (std::size_t i = 0; i < buffers.size(); ++i) {
            buffers[i] = BufferType(CoordBox<DIM>(Coord<DIM>(), bufferDim), curGrid->getEdgeCell(), curGrid->getEdgeCell());
        }
        LOG(DBG, "created " << buffers.size() << " buffers");

        generateFrames();
        nanoStep = 0;
    }

    virtual ~CacheBlockingSimulator()
    {
        delete newGrid;
        delete curGrid;
    }

    virtual void step()
    {
        // fixme
    }

    virtual void run()
    {
        initializer->grid(curGrid);
        stepNum = initializer->startStep();
        nanoStep = 0;

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                *getGrid(),
                getStep(),
                WRITER_INITIALIZED);
        }

        for (stepNum = initializer->startStep();
             stepNum < initializer->maxSteps();) {
            hop();
        }

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                *getGrid(),
                getStep(),
                WRITER_ALL_DONE);
        }
    }

    virtual const GridType *getGrid()
    {
        return curGrid;
    }

private:
    using MonolithicSimulator<CELL>::initializer;
    using MonolithicSimulator<CELL>::steerers;
    using MonolithicSimulator<CELL>::stepNum;
    using MonolithicSimulator<CELL>::writers;
    using MonolithicSimulator<CELL>::getStep;

    GridType *curGrid;
    GridType *newGrid;
    std::vector<BufferType> buffers;
    unsigned pipelineLength;
    Coord<DIM - 1> wavefrontDim;
    Grid<WavefrontFrames> frames;
    unsigned nanoStep;

    void generateFrames()
    {
        Coord<DIM> gridDim = initializer->gridBox().dimensions;

        Coord<DIM - 1> framesDim;
        for (int i = 0; i < (DIM - 1); ++i) {
            framesDim[i] = gridDim[i] / wavefrontDim[i];
            if ((gridDim[i] % wavefrontDim[i]) != 0) {
                framesDim[i] += 1;
            }
        }
        frames.resize(framesDim);

        for (int y = 0; y < framesDim.y(); ++y) {
            for (int x = 0; x < framesDim.x(); ++x) {
                frames[Coord<2>(x, y)] = generateWavefrontFrames(
                    Coord<3>(x * wavefrontDim.x(),
                             y * wavefrontDim.y(),
                             0));

            }
        }
    }

    WavefrontFrames generateWavefrontFrames(const Coord<DIM> offset)
    {
        Coord<DIM> gridDim = initializer->gridBox().dimensions;
        Coord<DIM> wavefrontRegionDim;
        for (int i = 0; i < (DIM - 1); ++i) {
            wavefrontRegionDim[i] = wavefrontDim[i];
        }
        wavefrontRegionDim[DIM - 1] = gridDim[DIM - 1] - offset[DIM - 1];
        Region<DIM> region;
        region << CoordBox<DIM>(offset, wavefrontRegionDim);
        std::vector<Region<3> > regions(pipelineLength);
        regions[pipelineLength - 1] = region.expandWithTopology(
            0, gridDim, Topologies::Cube<3>::Topology());

        for (int i = static_cast<int>(pipelineLength - 2); i >= 0; --i) {
            unsigned index = static_cast<unsigned>(i);
            regions[index] = regions[index + 1].expandWithTopology(
                1, gridDim, Topologies::Cube<3>::Topology());
        }

        std::size_t wavefrontLength = static_cast<std::size_t>(gridDim[DIM - 1] + 1);
        WavefrontFrames ret(wavefrontLength, std::vector<Region<DIM> >(pipelineLength));

        for (unsigned index = 0; index < wavefrontLength; ++index) {
            Coord<DIM> maskOrigin;
            maskOrigin[DIM - 1] = static_cast<int>(index);
            Topology::normalize(maskOrigin, initializer->gridDimensions());
            Coord<DIM> maskDim = gridDim;
            maskDim[DIM - 1] = 1;

            Region<DIM> mask;
            mask << CoordBox<DIM>(maskOrigin, maskDim);
            for (unsigned i = 0; i < pipelineLength; ++i) {
                ret[index][i] = regions[i] & mask;
            }
        }

        return ret;
    }

    void hop()
    {
        using std::swap;
        TimeTotal t(&chronometer);

        CoordBox<DIM - 1> frameBox = frames.boundingBox();
        // for (typename CoordBox<DIM -1>::Iterator waveIter = frameBox.begin();
        //      waveIter != frameBox.end();
        //      ++waveIter) {
        //     updateWavefront(*waveIter);
        // }

#pragma omp parallel for
        for (int y = 0; y < frameBox.dimensions.y(); ++y) {
            for (int x = 0; x < frameBox.dimensions.x(); ++x) {
                updateWavefront(&buffers[threadIndex()], Coord<2>(x, y));
            }
        }

        swap(curGrid, newGrid);
        unsigned curNanoStep = nanoStep + pipelineLength;
        stepNum += curNanoStep / NANO_STEPS;
        nanoStep = curNanoStep % NANO_STEPS;
    }

    void updateWavefront(BufferType *buffer, const Coord<DIM - 1>& wavefrontCoord)
    {
        LOG(INFO, "wavefrontCoord(" << wavefrontCoord << ")");
        buffer->setEdge(curGrid->getEdge());
        // buffer->fill(buffer->boundingBox(), curGrid->getEdgeCell());
        fixBufferOrigin(buffer, wavefrontCoord);

        unsigned index = 0;
        CoordBox<DIM> boundingBox = curGrid->boundingBox();
        unsigned maxIndex = static_cast<unsigned>(boundingBox.origin[DIM - 1] + boundingBox.dimensions[DIM - 1]);

        // fill pipeline
        unsigned maxLength = 2 * pipelineLength - 2;
        for (; index < maxLength; ++index) {
            unsigned lastStage = (index >> 1) + 1;
            pipelinedUpdate(buffer, wavefrontCoord, index, index, 0, lastStage);
        }

        // normal operation
        for (; index < maxIndex; ++index) {
            pipelinedUpdate(buffer, wavefrontCoord, index, index, 0, pipelineLength);
        }

        // let pipeline drain
        for (; index < (maxIndex + 2 * pipelineLength - 2); ++index) {
            unsigned firstStage = (index - maxIndex + 1) >> 1 ;
            pipelinedUpdate(buffer, wavefrontCoord, index, index, firstStage, pipelineLength);
        }
    }

    void fixBufferOrigin(BufferType *buffer, const Coord<DIM - 1>& frameCoord)
    {
        Coord<DIM> bufferOrigin;
        // fixme: wrong on boundary with Torus topology
        for (int d = 0; d < (DIM - 1); ++d) {
            bufferOrigin[d] = (std::max)(0, frameCoord[d] * wavefrontDim[d] - static_cast<int>(pipelineLength) + 1);
        }
        bufferOrigin[DIM - 1] = 0;
        buffer->setOrigin(bufferOrigin);
    }

    void pipelinedUpdate(
        BufferType *buffer,
        const Coord<DIM - 1>& frameCoord,
        unsigned globalIndex,
        unsigned localIndex,
        unsigned firstStage,
        unsigned lastStage)
    {
        LOG(DBG, "  pipelinedUpdate(frameCoord = " << frameCoord << ", globalIndex = " << globalIndex << ", localIndex = " << localIndex << ", firstStage = " << firstStage << ", lastStage = " << lastStage << ")");

        for (unsigned i = firstStage; i < lastStage; ++i) {
            bool firstIteration = (i == 0);
            bool lastIteration =  (i == (pipelineLength - 1));
            unsigned currentGlobalIndex = globalIndex - 2 * i;
            bool needsFlushing = (i == firstStage) &&
                (currentGlobalIndex >= static_cast<unsigned>(newGrid->getDimensions()[DIM - 1]));
            unsigned sourceIndex = firstIteration ? currentGlobalIndex : normalizeIndex(localIndex + 2 - 4 * i);
            unsigned targetIndex = lastIteration  ? currentGlobalIndex : normalizeIndex(localIndex + 0 - 4 * i);

            const Region<DIM>& updateFrame = frames[frameCoord][static_cast<int>(globalIndex) - 2 * i][i];
            unsigned curNanoStep = (nanoStep + i) % NANO_STEPS;

            if ( firstIteration &&  lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *curGrid, newGrid, curNanoStep);
            }

            if ( firstIteration && !lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *curGrid, buffer,  curNanoStep);
            }

            if (!firstIteration &&  lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *buffer,  newGrid, curNanoStep);
            }

            if (!firstIteration && !lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *buffer,  buffer,  curNanoStep);
            }
        }
    }

    template<class GRID1, class GRID2>
    void frameUpdate(
        bool needsFlushing,
        const Region<DIM>& updateFrame,
        unsigned sourceIndex,
        unsigned targetIndex,
        const GRID1& sourceGrid,
        GRID2 *targetGrid,
        unsigned curNanoStep)
    {
        LOG(DBG, "    frameUpdate("
            << needsFlushing << ", "
            << updateFrame.boundingBox() << ", "
            << sourceIndex << ", "
            << targetIndex <<  ")");

        if (needsFlushing) {
            // fixme: only works with cube topologies
            Coord<DIM> fillOrigin = buffers[threadIndex()].getOrigin();
            fillOrigin[DIM - 1] = static_cast<int>(targetIndex);
            Coord<DIM> fillDim = buffers[threadIndex()].getDimensions();
            fillDim[DIM - 1] = 1;

            // buffers[omp_get_thread_num()].fill(
            //     CoordBox<DIM>(fillOrigin, fillDim),
            //     buffers[omp_get_thread_num()].getEdgeCell());
        } else {
            Coord<DIM> sourceOffset;
            Coord<DIM> targetOffset;
            sourceOffset[DIM - 1] = static_cast<int>(sourceIndex);
            targetOffset[DIM - 1] = static_cast<int>(targetIndex);

            UpdateFunctor<CELL>()(updateFrame, sourceOffset, targetOffset, sourceGrid, targetGrid, curNanoStep);
        }
     }

    inline
    std::size_t threadIndex()
    {
        return static_cast<std::size_t>(omp_get_thread_num());
    }

    // wraps the index (for 3D this will be the Z coordinate) around
    // the buffer's dimension
    unsigned normalizeIndex(unsigned localIndex)
    {
        unsigned bufferSize = static_cast<unsigned>(buffers[0].getDimensions()[DIM - 1]);
        return (localIndex + bufferSize) % bufferSize;
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif

#endif
