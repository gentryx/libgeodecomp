#ifndef LIBGEODECOMP_PARALLELIZATION_STRIPINGSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_STRIPINGSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI

#include <algorithm>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/storage/serializationbuffer.h>

namespace LibGeoDecomp {

/**
 * This class aims at providing a very simple, but working parallel
 * simulation facility. It's not very modular, it's not fast, but it's
 * simple and it actually works.
 */
template<typename CELL_TYPE>
class StripingSimulator : public DistributedSimulator<CELL_TYPE>
{
public:
    friend class StripingSimulatorTest;
    friend class ParallelStripingSimulatorTest;

    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef LoadBalancer::WeightVec WeightVec;
    typedef LoadBalancer::LoadVec LoadVec;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, false, SupportsSoA>::Value GridType;
    typedef typename Steerer<CELL_TYPE>::SteererFeedback SteererFeedback;
    typedef typename SerializationBuffer<CELL_TYPE>::BufferType BufferType;
    typedef typename DistributedSimulator<CELL_TYPE>::InitPtr InitPtr;

    static const int DIM = Topology::DIM;
    static const bool WRAP_EDGES = Topology::template WrapsAxis<DIM - 1>::VALUE;

    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    using DistributedSimulator<CELL_TYPE>::chronometer;
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::getStep;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::stepNum;
    using DistributedSimulator<CELL_TYPE>::writers;
    using DistributedSimulator<CELL_TYPE>::gridDim;

    enum WaitTags {
        GENERAL,
        BALANCELOADS,
        GHOSTREGION
    };

    explicit StripingSimulator(
        Initializer<CELL_TYPE> *initializer,
        LoadBalancer *balancer = 0,
        unsigned loadBalancingPeriod = 1):
        DistributedSimulator<CELL_TYPE>(initializer),
        balancer(balancer),
        partitions(partition(initializer->gridDimensions()[DIM - 1], MPILayer().size())),
        loadBalancingPeriod(loadBalancingPeriod)
    {
        validateConstructorParams();

        initRegions(partitions);
        adaptBuffers();

        curStripe = new GridType(regionWithOuterGhosts);
        newStripe = new GridType(regionWithOuterGhosts);
        initSimulation();
    }

    explicit StripingSimulator(
        const InitPtr& initializer,
        LoadBalancer *balancer = 0,
        unsigned loadBalancingPeriod = 1):
        DistributedSimulator<CELL_TYPE>(initializer),
        balancer(balancer),
        partitions(partition(initializer->gridDimensions()[DIM - 1], MPILayer().size())),
        loadBalancingPeriod(loadBalancingPeriod)
    {
        validateConstructorParams();

        initRegions(partitions);
        adaptBuffers();

        curStripe = new GridType(regionWithOuterGhosts);
        newStripe = new GridType(regionWithOuterGhosts);
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

        handleInput(STEERER_NEXT_STEP);

        for (unsigned i = 0; i < NANO_STEPS; i++) {
            nanoStep(i);
        }

        ++stepNum;

        WriterEvent event = WRITER_STEP_FINISHED;
        if (stepNum == initializer->maxSteps()) {
            event = WRITER_ALL_DONE;
        }
        handleOutput(event);
    }

    /**
     * performs step() until the maximum number of steps is reached.
     */
    virtual void run()
    {
        initSimulation();
        setIORegions();
        handleOutput(WRITER_INITIALIZED);

        while (stepNum < initializer->maxSteps()) {
            step();
        }
    }

    inline unsigned getLoadBalancingPeriod() const
    {
        return loadBalancingPeriod;
    }

    std::vector<Chronometer> gatherStatistics()
    {
        return mpilayer.gather(chronometer, 0);
    }

private:
    MPILayer mpilayer;
    typename SharedPtr<LoadBalancer>::Type balancer;
    /**
     * we need to distinguish four types of rims:
     *   - the inner rim is sent to neighboring nodes (and lies whithin our own stripe)
     *   - the outer rim is received from them (and is appended beneath/above our stripe)
     *   - the lower rim is located at the lower edge of our stripe (where the absolute
     *     value of the y-coordinates are higher),
     *   - the upper rim conversely is at the upper edge (smaller coordinate values)
     *
     * we assume the inner and outer rims to be of the same size. "rim" is the
     * same as "ghost" and "ghost region".
     */
    GridType *curStripe;
    GridType *newStripe;
    Region<DIM> region;
    Region<DIM> regionWithOuterGhosts;
    Region<DIM> innerRegion;
    Region<DIM> innerGhostRegion;
    Region<DIM> remappedInnerRegion;
    Region<DIM> remappedInnerGhostRegion;
    std::map<int, Region<DIM> > innerGhostRegions;
    std::map<int, Region<DIM> > outerGhostRegions;
    std::map<int, BufferType>  sendBuffers;
    std::map<int, BufferType>  recvBuffers;

    // contains the start and stop rows for each node's stripe
    WeightVec partitions;
    unsigned loadBalancingPeriod;

    /**
     * these Regions will only be used by the UpdateFunctor. They
     * need to be remapped because some grid implementations may
     * use different IDs internally.
     */
    void remapUpdateRegions()
    {
        remappedInnerRegion = curStripe->remapRegion(innerRegion);
        remappedInnerGhostRegion = curStripe->remapRegion(innerGhostRegion);
    }

    /**
     * Send and receive buffers need to be resized whenever our ghost zones have changed
     */
    void adaptBuffers()
    {
        sendBuffers.clear();
        recvBuffers.clear();

        for (typename std::map<int, Region<DIM> >::iterator i = innerGhostRegions.begin();
             i != innerGhostRegions.end();
             ++i) {
            sendBuffers[i->first] = BufferType();
            SerializationBuffer<CELL_TYPE>::resize(&sendBuffers[i->first], i->second);
        }

        for (typename std::map<int, Region<DIM> >::iterator i = outerGhostRegions.begin();
             i != outerGhostRegions.end();
             ++i) {
            recvBuffers[i->first] = BufferType();
            SerializationBuffer<CELL_TYPE>::resize(&recvBuffers[i->first], i->second);
        }
    }

    void swapGrids()
    {
        using std::swap;
        swap(curStripe, newStripe);
    }

    /**
     * "partition()[i]" is the first row for which node i is
     * responsible, "partition()[i + 1] - 1" is the last one.
     */
    static WeightVec partition(unsigned gridHeight, unsigned size)
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

        // weird: GCC 4.7.3 refuses to let me use chronometer.ratio<Foo, Bar> directly.
        Chronometer& c = chronometer;
        double myRatio = c.template ratio<TimeCompute, TimeTotal>();
        chronometer.reset();
        LoadVec loads = mpilayer.gather(myRatio, 0);
        WeightVec newPartitionsSendBuffer;

        if (mpilayer.rank() == 0) {
            WeightVec oldWorkloads = partitionsToWorkloads(partitions);
            WeightVec newWorkloads = balancer->balance(oldWorkloads, loads);
            validateLoads(newWorkloads, oldWorkloads);
            newPartitionsSendBuffer = workloadsToPartitions(newWorkloads);

            for (int i = 0; i < mpilayer.size(); i++) {
                mpilayer.sendVec(&newPartitionsSendBuffer, i, BALANCELOADS);
            }
        }

        WeightVec oldPartitions = partitions;
        WeightVec newPartitions(partitions.size());
        mpilayer.recvVec(&newPartitions, 0, BALANCELOADS);
        mpilayer.wait(BALANCELOADS);

        redistributeGrid(oldPartitions, newPartitions);
    }

    void nanoStep(unsigned nanoStep)
    {
        TimeTotal t(&chronometer);

        // we wait for ghostregions "just in time" to overlap
        // communication with I/O (which occurs outside of
        // nanoStep()). actually we could stretch one cycle further to
        // have two concurrent communication requests per ghostregion,
        // but this would be a hell to code and debug.

        waitForGhostRegions(curStripe);
        recvOuterGhostRegion();
        updateInnerGhostRegion(nanoStep);
        sendInnerGhostRegion(newStripe);

        updateInside(nanoStep);
        swapGrids();
    }

    void waitForGhostRegions(GridType *stripe)
    {
        TimeCommunication t(&chronometer);

        bool hadPendingRequests = mpilayer.wait(GHOSTREGION);

        if (hadPendingRequests) {
            for (typename std::map<int, BufferType>::iterator i = recvBuffers.begin();
                 i != recvBuffers.end();
                 ++i) {

                stripe->loadRegion(i->second, outerGhostRegions[i->first]);
            }
        }
    }

    void setIORegions()
    {
        for(unsigned i = 0; i < writers.size(); i++) {
            writers[i]->setRegion(region);
        }

        for(unsigned i = 0; i < steerers.size(); i++) {
            steerers[i]->setRegion(region);
        }
    }

    void handleInput(SteererEvent event)
    {
        SteererFeedback feedback;
        // notify all registered Steerers
        waitForGhostRegions(curStripe);

        for(unsigned i = 0; i < steerers.size(); ++i) {
            if (stepNum % steerers[i]->getPeriod() == 0) {
                steerers[i]->nextStep(
                    curStripe,
                    regionWithOuterGhosts,
                    gridDim, getStep(),
                    event,
                    mpilayer.rank(),
                    true,
                    &feedback);
            }
        }

        // fixme: apply SteererFeedback!
    }

    void handleOutput(WriterEvent event)
    {
        for(unsigned i = 0; i < writers.size(); i++) {
            if ((event != WRITER_STEP_FINISHED) ||
                ((getStep() % writers[i]->getPeriod()) == 0)) {
                writers[i]->stepFinished(
                    *curStripe,
                    region,
                    gridDimensions(),
                    getStep(),
                    event,
                    mpilayer.rank(),
                    true);
            }
        }
    }

    /**
     * returns a bounding box with the same dimensions as the whole
     * grid, but the most significant dimension (z in the 3d case or y
     * in the 2d case) runs from start to end.
     */
    CoordBox<DIM> boundingBox(
        unsigned start,
        unsigned end)
    {
        int height = (std::max)((int)(end - start), 0);
        Coord<DIM> startCorner;
        Coord<DIM> dim = gridDimensions();
        startCorner[DIM - 1] = start;
        dim[DIM - 1] = height;

        return CoordBox<DIM>(startCorner, dim);
    }

    void updateRegion(const Region<DIM>& region, unsigned nanoStep)
    {
        UpdateFunctor<CELL_TYPE>()(
            region,
            Coord<DIM>(),
            Coord<DIM>(),
            *curStripe,
            newStripe,
            nanoStep);
    }

    /**
     * the methods below are just used to structurize the nanoStep() method.
     */
    void updateInnerGhostRegion(unsigned nanoStep)
    {
        TimeComputeGhost t(&chronometer);
        updateRegion(remappedInnerGhostRegion, nanoStep);
    }

    void recvOuterGhostRegion()
    {
        TimeCommunication t(&chronometer);

        for (typename std::map<int, BufferType>::iterator i = recvBuffers.begin();
             i != recvBuffers.end();
             ++i) {
            mpilayer.recv(
                i->second.data(),
                i->first,
                i->second.size(),
                GHOSTREGION,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType());

        }
    }

    void sendInnerGhostRegion(GridType *stripe)
    {
        TimeCommunication t(&chronometer);

        for (typename std::map<int, BufferType>::iterator i = sendBuffers.begin();
             i != sendBuffers.end();
             ++i) {
            stripe->saveRegion(&i->second, innerGhostRegions[i->first]);
            mpilayer.send(
                i->second.data(),
                i->first,
                i->second.size(),
                GHOSTREGION,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType());
        }
    }

    void updateInside(unsigned nanoStep)
    {
        TimeComputeInner t(&chronometer);
        updateRegion(remappedInnerRegion, nanoStep);
    }

    Region<DIM> fillRegion(int startRow, int endRow)
    {
        Region<DIM> ret;
        ret << boundingBox(startRow, endRow);
        return ret;
    }

    /**
     * resets various sizes, heights etc. according to
     * newPartitions, returns the new bounding box of the stripes. It
     * doesn't actually resize the stripes since different actions are
     * required during load balancing and initialization.
     */
    void initRegions(const WeightVec& newPartitions)
    {
        unsigned startRow = newPartitions[mpilayer.rank() + 0];
        unsigned endRow   = newPartitions[mpilayer.rank() + 1];
        region = fillRegion(startRow, endRow);
        {
            SharedPtr<Adjacency>::Type adjacency = initializer->getAdjacency(region);
            regionWithOuterGhosts = region.expandWithTopology(
                1,
                initializer->gridDimensions(),
                Topology(),
                *adjacency);
        }

        innerGhostRegion.clear();
        innerGhostRegions.clear();
        outerGhostRegions.clear();

        for (int i = 0; i < mpilayer.size(); ++i) {
            if (i != mpilayer.rank()) {
                startRow = newPartitions[i + 0];
                endRow   = newPartitions[i + 1];
                Region<DIM> otherRegion = fillRegion(startRow, endRow);
                SharedPtr<Adjacency>::Type adjacency = initializer->getAdjacency(otherRegion);
                Region<DIM> otherRegionExpanded = otherRegion.expandWithTopology(
                    1,
                    initializer->gridDimensions(),
                    Topology(),
                    *adjacency);

                Region<DIM> outerGhost = regionWithOuterGhosts & otherRegion;
                Region<DIM> innerGhost = region & otherRegionExpanded;

                if (!outerGhost.empty()) {
                    outerGhostRegions[i] = outerGhost;
                }
                if (!innerGhost.empty()) {
                    innerGhostRegions[i] = innerGhost;
                    innerGhostRegion += innerGhost;
                }
            }
        }

        innerRegion = region - innerGhostRegion;
        partitions = newPartitions;
    }

    /**
     * reset initial simulation state, useful after creation and at run()
     */
    void initSimulation()
    {
        chronometer.reset();

        delete curStripe;
        delete newStripe;
        curStripe = new GridType(regionWithOuterGhosts);
        newStripe = new GridType(regionWithOuterGhosts);
        initializer->grid(curStripe);
        initializer->grid(newStripe);
        stepNum = initializer->startStep();
        remapUpdateRegions();
    }

    WeightVec partitionsToWorkloads(const WeightVec& partitions) const
    {
        WeightVec ret(partitions.size() - 1);
        for (std::size_t i = 0; i < ret.size(); i++) {
            ret[i] = partitions[i + 1] - partitions[i];
        }
        return ret;
    }

    WeightVec workloadsToPartitions(const WeightVec& workloads) const
    {
        WeightVec ret(workloads.size() + 1);
        ret[0] = 0;
        for (std::size_t i = 0; i < workloads.size(); i++) {
            ret[i + 1] = ret[i] + workloads[i];
        }
        return ret;
    }

    void redistributeGrid(const WeightVec& oldPartitions,
                          const WeightVec& newPartitions)
    {
        waitForGhostRegions(curStripe);
        if (newPartitions == oldPartitions) {
            return;
        }
        initRegions(newPartitions);
        adaptBuffers();
        delete newStripe;
        newStripe = new GridType(regionWithOuterGhosts);
        initializer->grid(newStripe);

        // collect newStripe from others
        unsigned newStartRow = newPartitions[mpilayer.rank() + 0];
        unsigned newEndRow =   newPartitions[mpilayer.rank() + 1];
        std::vector<BufferType> receiveBuffers;
        std::vector<Region<DIM> > receiveRegions;
        receiveBuffers.reserve(newPartitions.size() - 1);

        for (std::size_t i = 0; i < newPartitions.size() - 1; ++i) {
            unsigned sourceStartRow = oldPartitions[i];
            unsigned sourceEndRow   = oldPartitions[i + 1];

            unsigned intersectionStart = (std::max)(newStartRow, sourceStartRow);
            unsigned intersectionEnd   = (std::min)(newEndRow,   sourceEndRow);

            if (intersectionEnd > intersectionStart) {
                Region<DIM> intersection = fillRegion(intersectionStart, intersectionEnd);
                receiveRegions << intersection;
                receiveBuffers << BufferType();
                SerializationBuffer<CELL_TYPE>::resize(&receiveBuffers.back(), intersection);

                mpilayer.recv(
                    receiveBuffers.back().data(),
                    i,
                    receiveBuffers.back().size(),
                    BALANCELOADS,
                    SerializationBuffer<CELL_TYPE>::cellMPIDataType());
            }
        }

        // send curStripe to others
        unsigned oldStartRow = oldPartitions[mpilayer.rank() + 0];
        unsigned oldEndRow =   oldPartitions[mpilayer.rank() + 1];
        std::vector<BufferType> sendBuffers;
        sendBuffers.reserve(newPartitions.size() - 1);

        for (std::size_t i = 0; i < newPartitions.size() - 1; ++i) {
            unsigned targetStartRow = newPartitions[i];
            unsigned targetEndRow   = newPartitions[i + 1];

            unsigned intersectionStart = (std::max)(oldStartRow, targetStartRow);
            unsigned intersectionEnd   = (std::min)(oldEndRow,   targetEndRow);

            if (intersectionEnd > intersectionStart) {
                Region<DIM> intersection = fillRegion(intersectionStart, intersectionEnd);
                sendBuffers << BufferType();
                SerializationBuffer<CELL_TYPE>::resize(&sendBuffers.back(), intersection);
                curStripe->saveRegion(&sendBuffers.back(), intersection);
                mpilayer.send(
                    sendBuffers.back().data(),
                    i,
                    sendBuffers.back().size(),
                    BALANCELOADS,
                    SerializationBuffer<CELL_TYPE>::cellMPIDataType());
            }
        }

        mpilayer.wait(BALANCELOADS);

        for (std::size_t i = 0; i < receiveBuffers.size(); ++i) {
            newStripe->loadRegion(receiveBuffers[i], receiveRegions[i]);
        }

        delete curStripe;
        curStripe = new GridType(regionWithOuterGhosts);
        initializer->grid(curStripe);
        swapGrids();
        sendInnerGhostRegion(curStripe);
        recvOuterGhostRegion();

        remapUpdateRegions();
    }

    /**
     * ensures that newLoads and oldLoads have the same size and that
     * the sums of their elements match.
     */
    void validateLoads(const WeightVec& newLoads, const WeightVec& oldLoads) const
    {
        if (newLoads.size() != oldLoads.size() ||
            sum(newLoads) != sum(oldLoads)) {
            throw std::invalid_argument(
                    "newLoads and oldLoads do not maintain invariance");
        }
    }

    inline Coord<DIM> gridDimensions() const
    {
        return initializer->gridDimensions();
    }

    void validateConstructorParams()
    {
        if (loadBalancingPeriod  < 1) {
            throw std::invalid_argument(
                "loadBalancingPeriod ( " + StringOps::itoa(loadBalancingPeriod) +
                ") must be positive");
        }

        // node 0 needs a (central) LoadBalancer...
        if ((mpilayer.rank() == 0) && (balancer.get() == 0)) {
            throw std::invalid_argument(
                "Rank " + StringOps::itoa(mpilayer.rank()) +
                "(Root) needs a non-empty LoadBalancer");
        }
        // ...while the others shouldn't have one (they rely on the central one).
        if ((mpilayer.rank() != 0) && (balancer.get() != 0)) {
            throw std::invalid_argument(
                "Rank " + StringOps::itoa(mpilayer.rank()) +
                "(Non-Root) needs an empty LoadBalancer");
        }
    }
};

}

#endif
#endif
