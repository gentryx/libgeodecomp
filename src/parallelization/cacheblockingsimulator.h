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
        int pipelineLength,
        const Coord<DIM - 1>& wavefrontDim) :
        MonolithicSimulator<CELL>(initializer),
        buffers(omp_get_max_threads()),
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
            bufferDim[i] = wavefrontDim[i] + 2 * pipelineLength - 2;
        }
        bufferDim[DIM - 1] = pipelineLength * 4 - 4;

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
    int pipelineLength;
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

        for (int i = pipelineLength - 2; i >= 0; --i) {
            regions[i] = regions[i + 1].expandWithTopology(
                1, gridDim, Topologies::Cube<3>::Topology());
        }

        int wavefrontLength = gridDim[DIM - 1] + 1;
        WavefrontFrames ret(wavefrontLength, std::vector<Region<DIM> >(pipelineLength));

        for (int index = 0; index < wavefrontLength; ++index) {
            Coord<DIM> maskOrigin;
            maskOrigin[DIM - 1] = index;
            Topology::normalize(maskOrigin, initializer->gridDimensions());
            Coord<DIM> maskDim = gridDim;
            maskDim[DIM - 1] = 1;

            Region<DIM> mask;
            mask << CoordBox<DIM>(maskOrigin, maskDim);
            for (int i = 0; i < pipelineLength; ++i) {
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
                updateWavefront(&buffers[omp_get_thread_num()], Coord<2>(x, y));
            }
        }

        swap(curGrid, newGrid);
        int curNanoStep = nanoStep + pipelineLength;
        stepNum += curNanoStep / NANO_STEPS;
        nanoStep = curNanoStep % NANO_STEPS;
    }

    void updateWavefront(BufferType *buffer, const Coord<DIM - 1>& wavefrontCoord)
    {
        LOG(INFO, "wavefrontCoord(" << wavefrontCoord << ")");
        buffer->setEdge(curGrid->getEdge());
        buffer->fill(buffer->boundingBox(), curGrid->getEdgeCell());
        fixBufferOrigin(buffer, wavefrontCoord);

        int index = 0;
        CoordBox<DIM> boundingBox = curGrid->boundingBox();
        int maxIndex = boundingBox.origin[DIM - 1] + boundingBox.dimensions[DIM - 1];

        // fill pipeline
        for (; index < 2 * pipelineLength - 2; ++index) {
            int lastStage = (index >> 1) + 1;
            pipelinedUpdate(buffer, wavefrontCoord, index, index, 0, lastStage);
        }

        // normal operation
        for (; index < maxIndex; ++index) {
            pipelinedUpdate(buffer, wavefrontCoord, index, index, 0, pipelineLength);
        }

        // let pipeline drain
        for (; index < (maxIndex + 2 * pipelineLength - 2); ++index) {
            int firstStage = (index - maxIndex + 1) >> 1 ;
            pipelinedUpdate(buffer, wavefrontCoord, index, index, firstStage, pipelineLength);
        }
    }

    void fixBufferOrigin(BufferType *buffer, const Coord<DIM - 1>& frameCoord)
    {
        Coord<DIM> bufferOrigin;
        // fixme: wrong on boundary with Torus topology
        for (int d = 0; d < (DIM - 1); ++d) {
            bufferOrigin[d] = std::max(0, frameCoord[d] * wavefrontDim[d] - pipelineLength + 1);
        }
        bufferOrigin[DIM - 1] = 0;
        buffer->setOrigin(bufferOrigin);
    }

    void pipelinedUpdate(
        BufferType *buffer,
        const Coord<DIM - 1>& frameCoord,
        int globalIndex,
        int localIndex,
        int firstStage,
        int lastStage)
    {
        LOG(DBG, "  pipelinedUpdate(frameCoord = " << frameCoord << ", globalIndex = " << globalIndex << ", localIndex = " << localIndex << ", firstStage = " << firstStage << ", lastStage = " << lastStage << ")");

        for (int i = firstStage; i < lastStage; ++i) {
            bool firstIteration = (i == 0);
            bool lastIteration =  (i == (pipelineLength - 1));
            int currentGlobalIndex = globalIndex - 2 * i;
            bool needsFlushing = (i == firstStage) &&
                (currentGlobalIndex >= newGrid->getDimensions()[DIM - 1]);
            int sourceIndex = firstIteration ? currentGlobalIndex : normalizeIndex(localIndex + 2 - 4 * i);
            int targetIndex = lastIteration  ? currentGlobalIndex : normalizeIndex(localIndex + 0 - 4 * i);

            const Region<DIM>& updateFrame = frames[frameCoord][globalIndex - 2 * i][i];
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
        int sourceIndex,
        int targetIndex,
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
            Coord<DIM> fillOrigin = buffers[omp_get_thread_num()].getOrigin();
            fillOrigin[DIM - 1] = targetIndex;
            Coord<DIM> fillDim = buffers[omp_get_thread_num()].getDimensions();
            fillDim[DIM - 1] = 1;

            buffers[omp_get_thread_num()].fill(
                CoordBox<DIM>(fillOrigin, fillDim),
                buffers[omp_get_thread_num()].getEdgeCell());
        } else {
            Coord<DIM> sourceOffset;
            Coord<DIM> targetOffset;
            sourceOffset[DIM - 1] = sourceIndex;
            targetOffset[DIM - 1] = targetIndex;

            UpdateFunctor<CELL>()(updateFrame, sourceOffset, targetOffset, sourceGrid, targetGrid, curNanoStep);
        }
     }

    // wraps the index (for 3D this will be the Z coordinate) around
    // the buffer's dimension
    int normalizeIndex(int localIndex)
    {
        int bufferSize = buffers[0].getDimensions()[DIM - 1];
        return (localIndex + bufferSize) % bufferSize;
    }
};

}

#endif

#endif
