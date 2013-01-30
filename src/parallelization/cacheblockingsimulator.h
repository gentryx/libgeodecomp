#ifndef _libgeodecomp_parallelization_serialsimulator_h_
#define _libgeodecomp_parallelization_serialsimulator_h_

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

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
    typedef typename CELL::Topology Topology;
    typedef typename Topologies::NDimensional<Topologies::NDimensional<Topologies::NDimensional<Topologies::ZeroDimensional, Topology::ParentTopology::ParentTopology::WRAP_EDGES>, Topology::ParentTopology::WRAP_EDGES>, true> BufferTopology;
    typedef Grid<CELL, Topology> GridType;
    typedef DisplacedGrid<CELL, BufferTopology> BufferType;
    typedef SuperVector<SuperVector<Region<3> > > WavefrontFrames;
    static const int DIM = Topology::DIMENSIONS;

    CacheBlockingSimulator(
        Initializer<CELL> *initializer,
        int pipelineLength,
        const Coord<DIM - 1>& wavefrontDim) : 
        MonolithicSimulator<CELL>(initializer),
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
        buffer = BufferType(CoordBox<DIM>(Coord<DIM>(), bufferDim), curGrid->getEdgeCell());

        generateFrames();
        nanoStep = 0;
    }
    
    ~CacheBlockingSimulator()
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
        // fixme
    }

    virtual const GridType *getGrid()
    {
        // fixme
        return 0;
    }

private:
    using MonolithicSimulator<CELL>::initializer;
    using MonolithicSimulator<CELL>::steerers;
    using MonolithicSimulator<CELL>::stepNum;
    using MonolithicSimulator<CELL>::writers;
    using MonolithicSimulator<CELL>::getStep;
    
    GridType *curGrid;
    GridType *newGrid;
    BufferType buffer;
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
        SuperVector<Region<3> > regions(pipelineLength);
        regions[pipelineLength - 1] = region.expandWithTopology(
            0, gridDim, Topologies::Cube<3>::Topology());
        
        for (int i = pipelineLength - 2; i >= 0; --i) {
            regions[i] = regions[i + 1].expandWithTopology(
                1, gridDim, Topologies::Cube<3>::Topology());
        }

        WavefrontFrames ret(gridDim[DIM - 1], SuperVector<Region<DIM> >(pipelineLength));

        for (int index = index; index < gridDim[DIM - 1]; ++index) {
            Coord<DIM> maskOrigin;
            maskOrigin[DIM - 1] = index;
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

    void pipelinedUpdate(
        const Coord<DIM - 1>& frameCoord, 
        int globalIndex, 
        int localIndex, 
        int firstStage, 
        int lastStage)
    {
        // fixme: do this before z-loop
        Coord<DIM> bufferOrigin;
        // fixme: wrong on boundary with Torus topology
        for (int d = 0; d < (DIM - 1); ++d) {
            bufferOrigin[d] = std::max(0, frameCoord[d] * wavefrontDim[d] - pipelineLength + 1);
        }
        bufferOrigin[DIM - 1] = 0;
        buffer.setOrigin(bufferOrigin);

        for (int i = firstStage; i < lastStage; ++i) {
            bool firstIteration = (i == 0);
            bool lastIteration =  (i == (pipelineLength - 1));
            int currentGlobalIndex = globalIndex - 2 * i;
            bool needsFlushing = (i == firstStage) &&
                (currentGlobalIndex >= newGrid->getDimensions()[DIM - 1]);
            int sourceIndex = firstIteration ? currentGlobalIndex : normalizeIndex(localIndex + 2 - 4 * i);
            int targetIndex = lastIteration  ? currentGlobalIndex : normalizeIndex(localIndex + 0 - 4 * i);
            const Region<DIM>& updateFrame = frames[frameCoord][globalIndex - 2 * i][i];
            unsigned curNanoStep = (nanoStep + i) % CELL::nanoSteps();

            std::string source = firstIteration ? "source" : "buffer";
            std::string target = lastIteration  ? "target" : "buffer";
            // std::cout << "update(grid: " 
            //           << source << "[" << std::setw(2) << sourceIndex << "] -> " 
            //           << target << "[" << std::setw(2) << targetIndex << "], time: " 
            //           << i << " -> " << (i + 1) << ", currentGlobalIndex: " 
            //           << currentGlobalIndex << ")\n";
            
            if ( firstIteration &&  lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *curGrid, newGrid, curNanoStep);
            }
            if ( firstIteration && !lastIteration) { 
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, *curGrid, &buffer, curNanoStep);
            }
            
            if (!firstIteration &&  lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, buffer, newGrid, curNanoStep);
            }
            if (!firstIteration && !lastIteration) {
                frameUpdate(needsFlushing, updateFrame, sourceIndex, targetIndex, buffer, &buffer, curNanoStep);
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
        if (needsFlushing) {
            Coord<DIM> fillOrigin = buffer.getOrigin();
            fillOrigin[DIM - 1] = targetIndex;
            Coord<DIM> fillDim = buffer.getDimensions();
            fillDim[DIM - 1] = 1;

            // std::cout << "flushing " << CoordBox<DIM>(fillOrigin, fillDim);
            buffer.fill(CoordBox<DIM>(fillOrigin, fillDim), buffer.getEdgeCell());
        } else {
            for (typename Region<DIM>::StreakIterator iter = updateFrame.beginStreak(); 
                 iter != updateFrame.endStreak(); ++iter) {
                Streak<DIM> sourceStreak = *iter;
                Coord<DIM> targetCoord  = iter->origin;
                sourceStreak.origin[DIM - 1] = sourceIndex;
                targetCoord[DIM - 1] = targetIndex;
                // fixme: fix nanostep
                UpdateFunctor<CELL>()(sourceStreak, targetCoord, sourceGrid, targetGrid, curNanoStep);
            }
            // std::cout << "  frame: " << updateFrame.boundingBox();
            // std::cout << "  sourceStreak: " << sourceStreak << "\n";
            // std::cout << "  targetStreak: " << targetStreak << "\n";
        }
        // std::cout << "  bufferOrigin: " << buffer.getOrigin() << "\n";
            
        // SuperVector<Line> *gridOld = &gridBuffer;
        // if (i == 0) {
        //     gridOld = &gridSource;
        // }

        // SuperVector<Line> *gridNew = &gridBuffer;
        // if (i == (pipelineLength - 1)) {
        //     gridNew = &gridTarget;
        // }
            
        // if ((globalIndex >= gridSource.size()) && (i == firstStage)) {
        //     (*gridNew)[targetIndex] = Line(-1, "X");
        // } else {
        //     (*gridNew)[targetIndex] = Line((*gridOld)[sourceIndex].offset, i + 1);
        // }

        // printGrid("source", gridSource);
        // printGrid("buffer", gridBuffer);
        // printGrid("target", gridTarget);
        // std::cout << "\n";
    }

    // normalize via topology, if at all?
    int normalizeIndex(int localIndex)
    {
        int bufferSize = buffer.getDimensions()[DIM - 1];
        return (localIndex + bufferSize) % bufferSize;
    }
};

}

#endif
