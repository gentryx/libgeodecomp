#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_h_
#define _libgeodecomp_parallelization_hiparsimulator_h_

#include <cmath>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/intersectingregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/innersetmarker.h>
#include <libgeodecomp/parallelization/hiparsimulator/rimmarker.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>
        
namespace LibGeoDecomp {
namespace HiParSimulator {

enum EventPoint {LOAD_BALANCING, END, PAUSE};
typedef SuperSet<EventPoint> EventSet;
typedef SuperMap<unsigned, EventSet> EventMap;

inline std::string eventToStr(const EventPoint& event) 
{
    switch(event) {
    case LOAD_BALANCING:
        return "LOAD_BALANCING";
    case END:
        return "END";
    case PAUSE:
        return "PAUSE";
    default:
        return "invalid";
    }
}

template<class CELL_TYPE, class PARTITION>
class HiParSimulator : public DistributedSimulator<CELL_TYPE>
{
    friend class HiParSimulatorTest;
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIMENSIONS;

    inline HiParSimulator(
        Initializer<CELL_TYPE> *_initializer,
        LoadBalancer *_balancer = 0,
        const unsigned& _loadBalancingPeriod = 1,
        const unsigned &_ghostZoneWidth = 1,
        MPI::Comm *_communicator = &MPI::COMM_WORLD) : 
        DistributedSimulator<CELL_TYPE>(_initializer),
        balancer(_balancer),
        loadBalancingPeriod(_loadBalancingPeriod * CELL_TYPE::nanoSteps()),
        ghostZoneWidth(_ghostZoneWidth),
        communicator(_communicator)
    {
        CoordBox<DIM> box = this->initializer->gridBox();
        updateGroup.reset(new UpdateGroup<CELL_TYPE, PARTITION>(
                              PARTITION(box.origin, box.dimensions),
                              initialWeights(box.dimensions.prod(), communicator->Get_size()),
                              0,
                              box,
                              ghostZoneWidth,
                              this->initializer,
                              communicator));
    }   

    // fixme: need test
    inline void run()
    {
        unsigned remainingNanoSteps = this->initializer->maxSteps() * 
            CELL_TYPE::nanoSteps() - nanoStepCounter;
        nanoStep(remainingNanoSteps);
    }

    // fixme: need test
    inline void step()
    {
        nanoStep(CELL_TYPE::nanoSteps());
    }

    // fixme: need test
    virtual void getGridFragment(
        const GridType **grid, 
        const Region<2> **validRegion) 
    {
        // *validRegion = &partitionManager.ownRegion();
    }

    // inline const DisplacedGrid<CELL_TYPE> *getDisplacedGrid()
    // {        
    //     return 0;
    //     // return regionStepper.getGrid();
    // }

private:
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    // fixme: need this?
    EventMap events;
    PartitionManager<DIM, Topology> partitionManager;
    MPI::Comm *communicator;
    boost::shared_ptr<UpdateGroup<CELL_TYPE, PARTITION> > updateGroup;

    SuperVector<long> initialWeights(const long& items, const long& size) const
    {    
        SuperVector<long> ret(size);
        long lastPos = 0;

        for (long i = 0; i < size; i++) {
            long currentPos = items * (i + 1) / size;
            ret[i] = currentPos - lastPos;
            lastPos = currentPos;
        }

        return ret;
    }

    inline void nanoStep(const unsigned& s)
    {
        updateGroup->update(s);

        // fixme: honor events here:
        // unsigned endNanoStep = nanoStepCounter + s;
        // events[endNanoStep].insert(PAUSE);
        
        // while (nanoStepCounter < endNanoStep) {
        //     std::pair<unsigned, EventSet> currentEvents = extractCurrentEvents();
        //     nanoStepCounter = currentEvents.first;
        //     handleEvents(currentEvents.second);
        // }
    }

//     inline std::pair<unsigned, EventSet> extractCurrentEvents()
//     {
//         EventMap::iterator curEventsPair = events.begin();
//         unsigned curStop  = curEventsPair->first;
//         EventSet curEvents = curEventsPair->second;
//         if (curStop < nanoStepCounter)
//             throw std::logic_error("Stale events found in event point queue");
//         events.erase(curEventsPair);
//         return std::make_pair(curStop, curEvents);
//     }


//     inline void handleEvents(const EventSet& curEvents)
//     {
//         //fixme: handle events
//         //             for (EventSet::iterator event = curEvents.begin(); event != curEvents.end(); ++event) 
//         //                 std::cout << "  nanoStep: " << nanoStepCounter << " got event: " << eventToStr(*event) << "\n";
//         //             if (curEvents.size() > 1)
//         //                 std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n";
//         //             std::cout << "\n";
            
//         if (curEvents.count(OUTPUT)) 
//             events[this->nextOutput(eventRepetitionHorizon)].insert(OUTPUT);
//         if (curEvents.count(LOAD_BALANCING))
//             events[this->nextLoadBalancing(eventRepetitionHorizon)].insert(LOAD_BALANCING);
//     }


//     inline void resetSimulation(const unsigned &ghostZoneWidth)
//     {
//         // this->resetEvents();
//         // this->resetRegions(ghostZoneWidth);
//     }

//     inline void resetRegions(const unsigned &ghostZoneWidth)
//     {
//         partitionManager.resetRegions(
//             CoordBox<2>(Coord<2>(), 
//                         this->initializer->gridDimensions()),
//             new VanillaRegionAccumulator<PARTITION>(
//             myPartition(),
//             myOffset(),
//             initialWeights()),
//             mpiLayer.rank(),
//             ghostZoneWidth);

//         SuperVector<CoordBox<2> > boundingBoxes(mpiLayer.size());
//         CoordBox<2> ownBoundingBox(partitionManager.ownRegion().boundingBox());
//         mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
//         partitionManager.resetGhostZones(boundingBoxes);
//         // fixme: care for validGhostZoneWidth
//     }

//     inline SuperVector<unsigned> initialWeights() const
//     {
//         SuperVector<unsigned> weights(mpiLayer.size());
//         if (mpiLayer.rank() == root) {
//             unsigned remainingCells = this->initializer->gridBox().size();
//             for (unsigned i = mpiLayer.size(); i > 0; --i) {
//                 unsigned curWeight = (unsigned)round((double)remainingCells / i);
//                 weights[i - 1] = curWeight;
//                 remainingCells -= curWeight;
//             }
//         }
//         mpiLayer.broadcastVector(&weights, root);
//         return weights;
//     }
    
//     inline void resetEvents()
//     {
//         nanoStepCounter = this->initializer->startStep() * CELL_TYPE::nanoSteps();
//         events.clear();
//         unsigned firstOutput        = ((int)ceil(nanoStepCounter / (double)outputPeriod))        * outputPeriod;
//         unsigned firstLoadBalancing = ((int)ceil(nanoStepCounter / (double)loadBalancingPeriod)) * loadBalancingPeriod;
//         for (int i = 0; i < eventRepetitionHorizon; ++i) {
//             events[firstOutput        + i * outputPeriod       ].insert(OUTPUT);
//             events[firstLoadBalancing + i * loadBalancingPeriod].insert(LOAD_BALANCING);
//         }
//         events[this->initializer->maxSteps() * CELL_TYPE::nanoSteps()].insert(SIMULATION_END);
//     }

//     inline unsigned nextOutput(const unsigned& horizon=1) const
//     {
//         return (nanoStepCounter / outputPeriod + horizon) * outputPeriod;
//     }    

//     inline unsigned nextLoadBalancing(const unsigned horizon=1) const
//     {
//         return (nanoStepCounter / loadBalancingPeriod + horizon) * loadBalancingPeriod;
//     }    

//     inline Region<2> allGatherGroupRegion() 
//     {
//         return allGatherGroupRegion(partitionManager.ownRegion());
//     }

//     inline Region<2> allGatherGroupRegion(const Region<2>& region) 
//     {
//         int streakNum = region.numStreaks();
//         SuperVector<int> streakNums(mpiLayer.allGather(streakNum));
//         SuperVector<Streak<2> > ownStreaks(region.toVector());
//         SuperVector<Streak<2> > allStreaks(mpiLayer.allGatherV(&ownStreaks[0], streakNums));
//         Region<2> ret;
//         for (SuperVector<Streak<2> >::iterator i = allStreaks.begin(); 
//              i != allStreaks.end(); ++i)
//             ret << *i;
//         return ret;
//     }

//     inline void registerOutgroupRegion(const unsigned& relativeLevel, const Region<2>& region)
//     {
// //         outgroupSteppers[relativeLevel].resetRegions(

// // IntersectingRegionAccumulator<PARTITION>(region, myPartition(), myOffset(), initialWeights());
//     }

//     inline void updateOutgroupRegion(const unsigned& relativeLevel, const unsigned& steps)
//     {
        
//     }

//     inline PARTITION myPartition() const
//     {
//         return PARTITION(Coord<2>(0, 0), 
//                          this->initializer->gridDimensions());
//     }

//     inline unsigned myOffset() const
//     {
//         return 0;
//     }

};

};
};

#endif
#endif
