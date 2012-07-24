#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_stripingsimulator_h_
#define _libgeodecomp_parallelization_stripingsimulator_h_

#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DefaultUpdateFunctor
{
public: 
    typedef typename CELL_TYPE::Topology Topology;

    template<int DIM, typename GRID_TYPE>
    void operator()(
        const Streak<DIM>& streak, 
        GRID_TYPE *curGrid, 
        GRID_TYPE *newGrid,
        const unsigned& nanoStep)
    {
        Coord<DIM> c = streak.origin;
        for (; c.x() < streak.endX; ++c.x()) {
            CoordMap<CELL_TYPE, Grid<CELL_TYPE, Topology> >  n  = 
                curGrid->getNeighborhood(c);
            (*newGrid)[c].update(n, nanoStep);
        }
    }
};

template<typename CELL_TYPE>
class StreakUpdateFunctor
{
public: 
    typedef typename CELL_TYPE::Topology Topology;

    template<int DIM, typename GRID_TYPE>
    void operator()(
        const Streak<DIM>& streak, 
        GRID_TYPE *curGrid, 
        GRID_TYPE *newGrid,
        const unsigned& nanoStep)
    {
        // fixme: this is bad...
        if (streak.origin.y() == 0   ||
            streak.origin.z() == 0   || 
            streak.origin.y() == 255 ||
            streak.origin.z() == 255)
            return;
        Streak<DIM> s = streak;
        Streak<DIM> tmpStreak = streak;

        // peel the first 1-2 iterations
        // fixme: this loop peeling at the start and end needs
        // rework. move to cell?
        if (streak.origin.x() & 1) {
            s.origin.x() += 1;
            tmpStreak.endX = tmpStreak.origin.x() + 1;
        } else {
            s.origin.x() += 2;
            tmpStreak.endX = tmpStreak.origin.x() + 2;
        }
        DefaultUpdateFunctor<CELL_TYPE>()(
            tmpStreak,
            curGrid,
            newGrid,
            nanoStep);

        int remainder = s.endX - s.origin.x();
        // "& -8" sets the three LSB to 0, thus always returning a
        // multiple of 8. subtract 1 in order to always peel one
        // iteration at the end.
        int length = (remainder - 1) & -8;
        CELL_TYPE *target = &((*newGrid)[s.origin + Coord<3>(0,  0,  0)]);

        CELL_TYPE *right  = &((*curGrid)[s.origin + Coord<3>(0,  0, -1)]);
        CELL_TYPE *top    = &((*curGrid)[s.origin + Coord<3>(0, -1,  0)]);
        CELL_TYPE *center = &((*curGrid)[s.origin + Coord<3>(0,  0,  0)]);
        CELL_TYPE *bottom = &((*curGrid)[s.origin + Coord<3>(0,  1,  0)]);
        CELL_TYPE *left   = &((*curGrid)[s.origin + Coord<3>(0,  0,  1)]);

        CELL_TYPE::update(
            target,
            right,
            top,
            center,
            bottom,
            left,
            length,
            nanoStep);

        // fixme: this loop peeling at the start and end needs
        // rework. move to cell?
        tmpStreak.origin.x() = s.origin.x() + length;
        tmpStreak.endX = streak.endX;
        // DefaultUpdateFunctor<CELL_TYPE>()(
        //     tmpStreak,
        //     curGrid,
        //     newGrid,
        //     nanoStep);
    }
};

template<typename CELL_TYPE>
class UpdateFunctor : public DefaultUpdateFunctor<CELL_TYPE>
{};

/**
 * This class aims at providing a very simple, but working parallel
 * simulation facility. It's not very modular, it's not fast, but it's
 * simple and it actually works.
 */
template<typename CELL_TYPE>
class StripingSimulator : public DistributedSimulator<CELL_TYPE>
{
    friend class StripingSimulatorTest;
    friend class ParallelStripingSimulatorTest;

public:
    typedef LoadBalancer::WeightVec WeightVec;
    typedef LoadBalancer::LoadVec LoadVec;
    typedef typename CELL_TYPE::Topology Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIMENSIONS;

    enum WaitTags {
        GENERAL, 
        BALANCELOADS, 
        GHOSTREGION_ALPHA,
        GHOSTREGION_BETA
    };

    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::writers;
    using DistributedSimulator<CELL_TYPE>::stepNum;

    StripingSimulator(
        Initializer<CELL_TYPE> *_initializer, 
        LoadBalancer *balancer = 0,
        const unsigned& _loadBalancingPeriod = 1,
        const MPI::Datatype& _cellMPIDatatype = Typemaps::lookup<CELL_TYPE>()): 
        DistributedSimulator<CELL_TYPE>(_initializer),
        balancer(balancer),
        loadBalancingPeriod(_loadBalancingPeriod),
        cellMPIDatatype(_cellMPIDatatype)
    {
        if (_loadBalancingPeriod  < 1) {
            throw std::invalid_argument(
                "loadBalancingPeriod ( " + StringConv::itoa(loadBalancingPeriod) + 
                ") must be positive");
        }
        
        // node 0 needs a (central) LoadBalancer...
        if (mpilayer.rank() == 0 && balancer == 0) {
            throw std::invalid_argument(
                "Rank " + StringConv::itoa(mpilayer.rank()) + 
                "(Root) needs a non-empty LoadBalancer");
        }
        // ...while the others shouldn't have one (they rely on the central one).
        if (mpilayer.rank() != 0 && balancer != 0) {
            throw std::invalid_argument(
                "Rank " + StringConv::itoa(mpilayer.rank()) + 
                "(Non-Root) needs an empty LoadBalancer");
        }

        int height = gridDimensions()[DIM - 1];
        partitions = partition(height, mpilayer.size());
        CoordBox<DIM> box = adaptDimensions(partitions);
        curStripe = new GridType(box);
        newStripe = new GridType(box);

        initSimulation();
    }

    virtual ~StripingSimulator()
    {
        mpilayer.waitAll();
        delete curStripe;
        delete newStripe;
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {   
        balanceLoad();
        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); i++)
            nanoStep(i);
        stepNum++;    
        handleOutput();
    }

    /**
     * performs step() until the maximum number of steps is reached.
     */
    virtual void run()
    {
        initSimulation();

        stepNum = initializer->startStep();
        for (unsigned i = 0; i < writers.size(); i++) {
            writers[i]->initialized();
        }

        while (stepNum < initializer->maxSteps()) {
            step();
        }
    
        for (unsigned i = 0; i < writers.size(); i++) {
            writers[i]->allDone();        
        }
    }

    virtual void getGridFragment(
        const GridBase<CELL_TYPE, DIM> **grid, 
        const Region<DIM> **validRegion)
    {
        *grid = curStripe;
        *validRegion = &region;
    }

    inline const unsigned& getLoadBalancingPeriod() const
    {
        return loadBalancingPeriod;
    }

private:
    MPILayer mpilayer;
    boost::shared_ptr<LoadBalancer> balancer;
    /**
     * we need to distinguish four types of rims: 
     *   - the inner rim is sent to neighboring nodes (and lies whithin our own 
     *     stripe)
     *   - the outer rim is received from them (and is appended beneath/above 
     *     our stripe)
     *   - the lower rim is located at the lower edge of our stripe (where the 
     *     absolute value of the y-coordinates are higher),
     *   - the upper rim conversely is at the upper edge (smaller coordinate 
     *     values) 
     *
     * we assume the inner and outer rims to be of the same size. "rim" is the 
     * same as "ghost" and "ghost region".
     */
    unsigned ghostHeightLower;
    unsigned ghostHeightUpper;
    GridType *curStripe;
    GridType *newStripe;
    Region<DIM> region;
    Region<DIM> innerRegion;
    Region<DIM> innerUpperGhostRegion;
    Region<DIM> innerLowerGhostRegion;
    Region<DIM> outerUpperGhostRegion;
    Region<DIM> outerLowerGhostRegion;
    // contains the start and stop rows for each node's stripe
    WeightVec partitions;
    unsigned loadBalancingPeriod;
    MPI::Datatype cellMPIDatatype;
    Chronometer chrono;

    void swapGrids()
    {
        std::swap(curStripe, newStripe);
    }

    /**
     * "partition()[i]" is the first row for which node i is
     * responsible, "partition()[i + 1] - 1" is the last one.
     */
    WeightVec partition(unsigned gridHeight, unsigned size) const
    {    
        WeightVec ret(size + 1);
        for (unsigned i = 0; i < size; i++) {
            ret[i] = gridHeight * i / size;
        }
        ret[size] = gridHeight;
        return ret;
    }

    /**
     * the methods below are just used to structurize the step() method. 
     */
    void balanceLoad()
    {
        if (stepNum % loadBalancingPeriod != 0) {
            return;
        }

        double ratio = chrono.nextCycle();
        LoadVec loads = mpilayer.gather(ratio, 0);

        if (mpilayer.rank() == 0) {
            WeightVec oldWorkloads = partitionsToWorkloads(partitions);
            WeightVec newWorkloads = balancer->balance(oldWorkloads, loads);
            validateLoads(newWorkloads, oldWorkloads);
            WeightVec newPartitions = workloadsToPartitions(newWorkloads);
            
            for (unsigned i = 0; i < mpilayer.size(); i++) 
                mpilayer.sendVec(&newPartitions, i, BALANCELOADS);
        }

        WeightVec oldPartitions = partitions;
        WeightVec newPartitions(partitions.size());
        mpilayer.recvVec(&newPartitions, 0, BALANCELOADS);
        mpilayer.wait(BALANCELOADS);
        
        redistributeGrid(oldPartitions, newPartitions);
    }

    void nanoStep(const unsigned& nanoStep)
    {
        // we wait for ghostregions "just in time" to overlap
        // communication with I/O (which occurs outside of
        // nanoStep()). actually we could stretch one cycle further to
        // have two concurrent communication requests per ghostregion,
        // but this would be a hell to code and debug.

        waitForGhostRegions();
        recvOuterGhostRegion(newStripe);
        updateInnerGhostRegion(nanoStep);
        sendInnerGhostRegion(newStripe);

        updateInside(nanoStep);
        swapGrids();
    }

    void waitForGhostRegions()
    {
        mpilayer.wait(GHOSTREGION_ALPHA);
        mpilayer.wait(GHOSTREGION_BETA);
    }

    void handleOutput()
    {
        for(unsigned i = 0; i < writers.size(); i++) {
            writers[i]->stepFinished();
        }
    }

    /**
     * returns a bounding box with the same dimensions as the whole
     * grid, but the most significant dimension (z in the 3d case or y
     * in the 2d case) runs from start to end.
     */ 
    CoordBox<DIM> boundingBox(
        const unsigned& start, 
        const unsigned& end)
    {
        int height = std::max((int)(end - start), 0);
        Coord<DIM> startCorner;
        Coord<DIM> dim = gridDimensions();
        startCorner[DIM - 1] = start;
        dim[DIM - 1] = height;

        return CoordBox<DIM>(startCorner, dim);
    }

    void updateRegion(const Region<DIM> &region, const unsigned& nanoStep)
    {
        chrono.tic();
        for (StreakIterator<DIM> i = region.beginStreak(); 
             i != region.endStreak(); 
             ++i) {
            UpdateFunctor<CELL_TYPE>()(
                *i,
                curStripe,
                newStripe,
                nanoStep);
        }

        chrono.toc();
    }

    /**
     * the methods below are just used to structurize the nanoStep() method. 
     */
    void updateInnerGhostRegion(const unsigned& nanoStep)
    {
        updateRegion(innerUpperGhostRegion, nanoStep);
        updateRegion(innerLowerGhostRegion, nanoStep);
    }

    void recvOuterGhostRegion(GridType *stripe)
    {    
        int upperNeighborRank = upperNeighbor();
        int lowerNeighborRank = lowerNeighbor();

        if (upperNeighborRank != -1) 
            mpilayer.recvUnregisteredRegion(
                stripe, 
                outerUpperGhostRegion, 
                upperNeighborRank, 
                GHOSTREGION_ALPHA,
                cellMPIDatatype);
        if (lowerNeighborRank != -1)
            mpilayer.recvUnregisteredRegion(
                stripe, 
                outerLowerGhostRegion, 
                lowerNeighborRank, 
                GHOSTREGION_BETA,
                cellMPIDatatype);
    }

    void sendInnerGhostRegion(GridType *stripe)
    {
        int upperNeighborRank = upperNeighbor();
        int lowerNeighborRank = lowerNeighbor();

        if (upperNeighborRank != -1) 
            mpilayer.sendUnregisteredRegion(
                stripe, 
                innerUpperGhostRegion, 
                upperNeighborRank, 
                GHOSTREGION_BETA,
                cellMPIDatatype);
        if (lowerNeighborRank != -1)
            mpilayer.sendUnregisteredRegion(
                stripe, 
                innerLowerGhostRegion, 
                lowerNeighborRank, 
                GHOSTREGION_ALPHA,
                cellMPIDatatype);
    }

    void updateInside(const unsigned& nanoStep)
    {
        updateRegion(innerRegion,  nanoStep);
    }

    Region<DIM> fillRegion(const int& startRow, const int& endRow)
    {
        Region<DIM> ret;
        ret << boundingBox(startRow, endRow);
        return ret;
    }

    void initRegions(const int& startRow, const int& endRow)
    {
        region      = fillRegion(startRow, endRow);
        innerRegion = fillRegion(startRow + ghostHeightUpper, endRow - ghostHeightLower);
        innerUpperGhostRegion = fillRegion(startRow, startRow +  ghostHeightUpper);
        innerLowerGhostRegion = fillRegion(endRow - ghostHeightLower, endRow);
        outerUpperGhostRegion = fillRegion(startRow - ghostHeightUpper, startRow);
        outerLowerGhostRegion = fillRegion(endRow, endRow + ghostHeightLower);
    }

    /**
     * reset initial simulation state, useful after creation and at run()
     */
    void initSimulation()
    {
        chrono.nextCycle();

        CoordBox<DIM> box = curStripe->boundingBox();
        curStripe->resize(box);
        newStripe->resize(box);
        initializer->grid(curStripe);
        newStripe->getEdgeCell() = curStripe->getEdgeCell();
        stepNum = 0;
    }

    /**
     * resets various sizes, heights etc. according to @a
     * newPartitions, returns the new bounding box of the stripes. It
     * doesn't actually resize the stripes since different actions are
     * required during load balancing and initialization.
     */
    CoordBox<DIM> adaptDimensions(const WeightVec& newPartitions)
    {
        unsigned startRow = 
            newPartitions[mpilayer.rank()    ];
        unsigned endRow =   
            newPartitions[mpilayer.rank() + 1];    

        // no need for ghostzones if zero-height stripe
        if (startRow == endRow) {
            ghostHeightUpper = 0;
            ghostHeightLower = 0;
        } else {
            if (Topology::WRAP_EDGES) {
                ghostHeightUpper = 1;
                ghostHeightLower = 1;
            } else {
                ghostHeightUpper = (startRow > 0? 1 : 0);
                ghostHeightLower = (endRow   < newPartitions.back()? 1 : 0);
            }
        }

        // no need for upper/lower ghost if only one node is present
        if (newPartitions.size() == 2) {
            ghostHeightUpper = 0;
            ghostHeightLower = 0;
        }

        initRegions(startRow, endRow);
        partitions = newPartitions;

        CoordBox<DIM> box = boundingBox(
            startRow - ghostHeightUpper, 
            endRow   + ghostHeightLower);
        return box;
    }

    int lowerNeighbor() const
    {
        int lowerNeighbor;

        if (Topology::WRAP_EDGES) {
            int size = mpilayer.size();
            lowerNeighbor = (size + mpilayer.rank() + 1) % size;
            while (lowerNeighbor != mpilayer.rank() && 
                   partitions[lowerNeighbor] == partitions[lowerNeighbor + 1])
                lowerNeighbor = (size + lowerNeighbor + 1) % size;
        } else {
             lowerNeighbor = mpilayer.rank() + 1;
            while (lowerNeighbor != mpilayer.size() && 
                   partitions[lowerNeighbor] == partitions[lowerNeighbor + 1])
                lowerNeighbor++;
            if (lowerNeighbor == mpilayer.size())
                lowerNeighbor = -1;
        }
        return lowerNeighbor;
    }

    int upperNeighbor() const
    {
        int upperNeighbor;

        if (Topology::WRAP_EDGES) {
            int size = mpilayer.size();
            upperNeighbor = (size + mpilayer.rank() - 1) % size;
            while (upperNeighbor != mpilayer.rank() && 
                   partitions[upperNeighbor] == partitions[upperNeighbor + 1])
                upperNeighbor = (size + upperNeighbor - 1) % size;   
        } else {
            upperNeighbor = mpilayer.rank() - 1;
            while (upperNeighbor >= 0 && 
                   partitions[upperNeighbor] == partitions[upperNeighbor + 1])
                upperNeighbor--;   
        }
        
        return upperNeighbor;
    }

    WeightVec partitionsToWorkloads(const WeightVec& partitions) const
    {
        WeightVec ret(partitions.size() - 1);
        for (unsigned i = 0; i < ret.size(); i++) {
            ret[i] = partitions[i + 1] - partitions[i];
        }
        return ret;
    }

    WeightVec workloadsToPartitions(const WeightVec& workloads) const
    {
        WeightVec ret(workloads.size() + 1);
        ret[0] = 0;
        for (unsigned i = 0; i < workloads.size(); i++) {
            ret[i + 1] = ret[i] + workloads[i];
        }
        return ret;
    }

    void redistributeGrid(const WeightVec& oldPartitions, 
                          const WeightVec& newPartitions)
    {
        waitForGhostRegions();
        if (newPartitions == oldPartitions) return;
        CoordBox<DIM> box = adaptDimensions(newPartitions);
        newStripe->resize(box);

        // collect newStripe from others
        unsigned newStartRow = 
            newPartitions[mpilayer.rank()    ];
        unsigned newEndRow =
            newPartitions[mpilayer.rank() + 1];
        for (int i = 0; i < newPartitions.size(); ++i) {
            unsigned sourceStartRow = oldPartitions[i];
            unsigned sourceEndRow   = oldPartitions[i + 1];

            unsigned intersectionStart = std::max(newStartRow, sourceStartRow);
            unsigned intersectionEnd   = std::min(newEndRow,   sourceEndRow);

            if (intersectionEnd > intersectionStart) {
                Region<DIM> intersection = fillRegion(intersectionStart, intersectionEnd);
                mpilayer.recvUnregisteredRegion(
                    newStripe,
                    intersection,
                    i,
                    BALANCELOADS,
                    cellMPIDatatype);
            }
        }

        // send curStripe to others
        unsigned oldStartRow = 
            oldPartitions[mpilayer.rank()    ];
        unsigned oldEndRow =   
            oldPartitions[mpilayer.rank() + 1];    
        for (int i = 0; i < newPartitions.size(); ++i) {
            unsigned targetStartRow = newPartitions[i];
            unsigned targetEndRow   = newPartitions[i + 1];

            unsigned intersectionStart = std::max(oldStartRow, targetStartRow);
            unsigned intersectionEnd   = std::min(oldEndRow,   targetEndRow);

            if (intersectionEnd > intersectionStart) {
                Region<DIM> intersection = fillRegion(intersectionStart, intersectionEnd);
                mpilayer.sendUnregisteredRegion(
                    curStripe,
                    intersection,
                    i,
                    BALANCELOADS,
                    cellMPIDatatype);
            }
        }
    
        mpilayer.wait(BALANCELOADS);
        curStripe->resize(box);
        swapGrids();
        sendInnerGhostRegion(curStripe);
        recvOuterGhostRegion(curStripe);
    }

    /**
     * ensures that newLoads and oldLoads have the same size and that
     * the sums of their elements match.
     */
    void validateLoads(const WeightVec& newLoads, const WeightVec& oldLoads) const
    {
        if (newLoads.size() != oldLoads.size() || 
            newLoads.sum() != oldLoads.sum()) {
            throw std::invalid_argument(
                    "newLoads and oldLoads do not maintain invariance");
        }
    }

    inline Coord<DIM> gridDimensions() const
    {
        return initializer->gridDimensions();
    }
};

}

#endif
#endif
