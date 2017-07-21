#ifndef LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>

#include <functional>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/storage/reorderingunstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <mutex>
#include <stdexcept>

namespace LibGeoDecomp {

namespace HPXDataFlowSimulatorHelpers {

/**
 * A draft for the event type to be passed to all cells in the future
 * -- instead of a plain integer representing the nano step.
 */
class UpdateEvent
{
public:
    inline
    UpdateEvent(int nanoStep, int step) :
        myNanoStep(nanoStep),
        myStep(step)
    {}

    inline int nanoStep() const
    {
        return myNanoStep;
    }

    inline int step() const
    {
        return myStep;
    }

private:
    int myNanoStep;
    int myStep;
};

/**
 * A lightweight implementation of the Neighborhood concept, tailored
 * for HPX dataflow.
 */
template<typename MESSAGE>
class Neighborhood
{
public:
    inline Neighborhood(
        int targetGlobalNanoStep,
        const std::vector<int> *messageNeighborIDs,
        std::vector<hpx::shared_future<MESSAGE> > *messagesFromNeighbors,
        const std::map<int, hpx::id_type> *remoteIDs) :
        targetGlobalNanoStep(targetGlobalNanoStep),
        messageNeighborIDs(messageNeighborIDs),
        messagesFromNeighbors(messagesFromNeighbors),
        remoteIDs(remoteIDs)
    {}

    inline
    const std::vector<int>& neighbors() const
    {
        return *messageNeighborIDs;
    }

    inline
    const MESSAGE& operator[](int index) const
    {
        std::vector<int>::const_iterator i = std::find(messageNeighborIDs->begin(), messageNeighborIDs->end(), index);
        if (i == messageNeighborIDs->end()) {
            throw std::logic_error("ID not found for incoming messages");
        }

        return (*messagesFromNeighbors)[i - messageNeighborIDs->begin()].get();
    }

    /**
     * Send a message to the cell known by the given ID. Odd: move
     * semantics seem to make this code run slower according to our
     * performance tests.
     * The send function is overloaded: the message can either be
     * an rvalue (MESSAGE&& message) or a const reference
     * (const MESSAGE& message).  In the first case, hpx will
     * not make a copy of the message.
     */
    inline
    void send(int remoteCellID,  MESSAGE&& message)
    {
        std::map<int, hpx::id_type>::const_iterator iter = remoteIDs->find(remoteCellID);
        if (iter == remoteIDs->end()) {
            throw std::logic_error("ID not found for outgoing messages");
        }

        sentNeighbors << remoteCellID;

        hpx::apply(
            typename HPXReceiver<MESSAGE>::receiveAction(),
            iter->second,
            targetGlobalNanoStep,
            std::move(message));
    }

    inline
    void send(int remoteCellID,  const MESSAGE& message)
    {
        std::map<int, hpx::id_type>::const_iterator iter = remoteIDs->find(remoteCellID);
        if (iter == remoteIDs->end()) {
            throw std::logic_error("ID not found for outgoing messages");
        }

        sentNeighbors << remoteCellID;

        hpx::apply(
            typename HPXReceiver<MESSAGE>::receiveAction(),
            iter->second,
            targetGlobalNanoStep,
            message);
    }

    inline
    void sendEmptyMessagesToUnnotifiedNeighbors()
    {
        for (int neighbor: *messageNeighborIDs) {
            if (std::find(sentNeighbors.begin(), sentNeighbors.end(), neighbor) == sentNeighbors.end()) {
                send(neighbor, MESSAGE());
            }
        }
    }

private:
    int targetGlobalNanoStep;
    const std::vector<int> *messageNeighborIDs;
    std::vector<hpx::shared_future<MESSAGE> > *messagesFromNeighbors;
    const std::map<int, hpx::id_type> *remoteIDs;
    // fixme: make this optional!
    std::vector<int> sentNeighbors;
};

template<typename CELL, typename MESSAGE>
class CellComponent : public hpx::components::component_base<CellComponent<CELL, MESSAGE> >
{
public:
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL>::VALUE;
    typedef typename APITraits::SelectMessageType<CELL>::Value MessageType;
    typedef ReorderingUnstructuredGrid<UnstructuredGrid<CELL> > GridType;

    explicit CellComponent(
            const std::string& basename = "",
            typename SharedPtr<GridType>::Type grid = 0,
            int id = -1,
            const std::vector<int>& neighbors = std::vector<int>())
      : basename(basename),
        neighbors(neighbors),
        grid(grid),
        id(id)
    {
        for (auto&& neighbor: neighbors) {
            std::string linkName = endpointName(basename, neighbor, id);
            receivers[neighbor] = HPXReceiver<MESSAGE>::make(linkName).get();
        }
    }

    void setupRemoteReceiverIDs()
    {
        std::vector<hpx::future<void> > remoteIDFutures;
        remoteIDFutures.reserve(neighbors.size());
        hpx::lcos::local::spinlock mutex;

        for (auto i = neighbors.begin(); i != neighbors.end(); ++i) {
            std::string linkName = endpointName(basename, id, *i);

            int neighbor = *i;
            remoteIDFutures << HPXReceiver<MessageType>::find(linkName).then(
                [&mutex, neighbor, this](hpx::shared_future<hpx::id_type> remoteIDFuture)
                {
                    std::lock_guard<hpx::lcos::local::spinlock> l(mutex);
                    remoteIDs[neighbor] = remoteIDFuture.get();
                });
        }

        hpx::when_all(remoteIDFutures).get();
    }

    hpx::shared_future<void> setupDataflow(hpx::shared_future<void> lastTimeStepFuture, int startStep, int endStep)
    {
        if (startStep == 0) {
            setupRemoteReceiverIDs();
        }

        // fixme: add steerer/writer interaction
        for (int step = startStep; step < endStep; ++step) {
            for (std::size_t nanoStep = 0; nanoStep < NANO_STEPS; ++nanoStep) {
                int globalNanoStep = step * NANO_STEPS + nanoStep;

                std::vector<hpx::shared_future<MessageType> > receiveMessagesFutures;
                receiveMessagesFutures.reserve(neighbors.size());

                for (auto&& neighbor: neighbors) {
                    if (globalNanoStep > 0) {
                        receiveMessagesFutures << receivers[neighbor]->get(globalNanoStep);
                    } else {
                        receiveMessagesFutures << hpx::make_ready_future(MessageType());
                    }
                }

                hpx::shared_future<void> thisTimeStepFuture = hpx::dataflow(
                    hpx::launch::async,
                    &HPXDataFlowSimulatorHelpers::CellComponent<CELL, MessageType>::update,
                    this,
                    neighbors,
                    std::move(receiveMessagesFutures),
                    lastTimeStepFuture,
                    nanoStep,
                    step);

                using std::swap;
                swap(thisTimeStepFuture, lastTimeStepFuture);
            }
        }

        return lastTimeStepFuture;
    }
    HPX_DEFINE_COMPONENT_ACTION(CellComponent, setupDataflow);

    void update(
        const std::vector<int>& neighbors,
        std::vector<hpx::shared_future<MESSAGE> >&& inputFutures,
        // Unused, just here to ensure correct ordering of updates per cell:
        const hpx::shared_future<void>& lastTimeStepReady,
        int nanoStep,
        int step)
    {
        int targetGlobalNanoStep = step * NANO_STEPS + nanoStep + 1;
        Neighborhood<MESSAGE> hood(targetGlobalNanoStep, &neighbors, &inputFutures, &remoteIDs);

        UpdateEvent event(nanoStep, step);

        cell()->update(hood, event);
        hood.sendEmptyMessagesToUnnotifiedNeighbors();
    }

private:
    std::string basename;
    std::vector<int> neighbors;
    typename SharedPtr<GridType>::Type grid;
    int id;
    std::map<int, std::shared_ptr<HPXReceiver<MESSAGE> > > receivers;
    std::map<int, hpx::id_type> remoteIDs;

    static std::string endpointName(const std::string& basename, int sender, int receiver)
    {
        return "HPXDataflowSimulatorEndPoint_" +
            basename +
            "_" +
            StringOps::itoa(sender) +
            "_to_" +
            StringOps::itoa(receiver);
    }

    CELL *cell()
    {
        return grid->data();
    }
};

}

#define REGISTER_CELLCOMPONENT(CELL, MESSAGE, name)                           \
    typedef ::hpx::components::component<                                     \
        LibGeoDecomp::HPXDataFlowSimulatorHelpers::CellComponent<CELL, MESSAGE> > \
            BOOST_PP_CAT(__cellcomponent_, name);                             \
    HPX_REGISTER_ACTION(BOOST_PP_CAT(__cellcomponent_, name)::setupDataflow_action, \
        BOOST_PP_CAT(__cellcomponent_setupDataflow_action_, name));           \
    HPX_REGISTER_COMPONENT(BOOST_PP_CAT(__cellcomponent_, name))              \
/**/

/**
 * Experimental Simulator based on (surprise surprise) HPX' dataflow
 * operator. Primary use case (for now) is DGSWEM.
 */
template<typename CELL, typename PARTITION = UnstructuredStripingPartition>
class HPXDataflowSimulator : public DistributedSimulator<CELL>
{
public:
    typedef typename APITraits::SelectMessageType<CELL>::Value MessageType;
    typedef DistributedSimulator<CELL> ParentType;
    typedef typename DistributedSimulator<CELL>::Topology Topology;
    typedef PartitionManager<Topology> PartitionManagerType;
    using DistributedSimulator<CELL>::NANO_STEPS;
    using DistributedSimulator<CELL>::initializer;

    /**
     * basename will be added to IDs for use in AGAS lookup, so for
     * each simulation all localities need to use the same basename,
     * but if you intent to run multiple different simulations in a
     * single program, either in parallel or sequentially, you'll need
     * to use a different basename.
     */
    inline HPXDataflowSimulator(
        Initializer<CELL> *initializer,
        const std::string& basename,
        int chunkSize = 5) :
        ParentType(initializer),
        basename(basename),
        chunkSize(chunkSize)
    {}

    void step()
    {
        throw std::logic_error("HPXDataflowSimulator::step() not implemented");
    }

    long currentNanoStep() const
    {
        throw std::logic_error("HPXDataflowSimulator::currentNanoStep() not implemented");
        return 0;
    }

    void balanceLoad()
    {
        throw std::logic_error("HPXDataflowSimulator::balanceLoad() not implemented");
    }

    void run()
    {
        Region<1> localRegion;
        CoordBox<1> box = initializer->gridBox();
        std::size_t rank = hpx::get_locality_id();
        std::size_t numLocalities = hpx::get_num_localities().get();

        std::vector<double> rankSpeeds(numLocalities, 1.0);
        std::vector<std::size_t> weights = LoadBalancer::initialWeights(
            box.dimensions.prod(),
            rankSpeeds);

        Region<1> globalRegion;
        globalRegion << box;

        typename SharedPtr<PARTITION>::Type partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                weights,
                initializer->getAdjacency(globalRegion)));

        PartitionManager<Topology> partitionManager;
        partitionManager.resetRegions(
            initializer,
            box,
            partition,
            rank,
            1);

        localRegion = partitionManager.ownRegion();
        SharedPtr<Adjacency>::Type adjacency = initializer->getAdjacency(localRegion);

        typedef HPXDataFlowSimulatorHelpers::CellComponent<CELL, MessageType> ComponentType;
        typedef typename ComponentType::GridType GridType;
        typedef hpx::components::client<ComponentType> CellClient;

        std::map<int, CellClient> components;
        std::vector<int> neighbors;

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            int id = i->x();
            CoordBox<1> singleCellBox(Coord<1>(id), Coord<1>(1));
            typename SharedPtr<GridType>::Type grid(new GridType(singleCellBox));
            initializer->grid(&*grid);

            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);
            components[i->x()] = hpx::local_new<CellClient>(
                basename,
                grid,
                i->x(),
                neighbors);
        }

        // HPX Reset counters:
        hpx::reset_active_counters();

        typedef hpx::shared_future<void> UpdateResultFuture;
        typedef std::vector<UpdateResultFuture> TimeStepFutures;
        TimeStepFutures lastTimeStepFutures(localRegion.size(), hpx::make_ready_future());
        TimeStepFutures nextTimeStepFutures;
        nextTimeStepFutures.reserve(localRegion.size());
        int maxTimeSteps = initializer->maxSteps();

        // HPX Sliding semaphore
        // allow larger look-ahead for dataflow generation to better
        // overlap calculation and computation:
        int lookAheadDistance = 2 * chunkSize;
        hpx::lcos::local::sliding_semaphore semaphore(lookAheadDistance);

        for (int startStep = 0; startStep < maxTimeSteps; startStep += chunkSize) {
            int endStep = (std::min)(maxTimeSteps, startStep + chunkSize);
            std::size_t index = 0;
            for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
                nextTimeStepFutures <<
                    hpx::async(typename ComponentType::setupDataflow_action(),
                        components[i->x()], lastTimeStepFutures[index],
                        startStep, endStep);
                ++index;
            }

            nextTimeStepFutures[0].then(
                [&semaphore, startStep](hpx::shared_future<void>) {
                    // inform semaphore about new lower limit
                    semaphore.signal(startStep);
                });

            semaphore.wait(startStep);
            using std::swap;
            swap(lastTimeStepFutures, nextTimeStepFutures);
            nextTimeStepFutures.clear();
        }

        hpx::when_all(lastTimeStepFutures).get();
    }

    std::vector<Chronometer> gatherStatistics()
    {
        // fixme
        return std::vector<Chronometer>();
    }

private:
    std::string basename;
    int chunkSize;
};

}

#endif

#endif
