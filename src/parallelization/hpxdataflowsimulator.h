#ifndef LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/parallelization/hierarchicalsimulator.h>
#include <stdexcept>

namespace LibGeoDecomp {

namespace HPXDataFlowSimulatorHelpers {

template<typename MESSAGE>
class Neighborhood
{
public:
    // fixme: move semantics
    inline Neighborhood(
        std::vector<int> messageNeighborIDs,
        std::vector<hpx::shared_future<MESSAGE> > messagesFromNeighbors,
        const std::map<int, hpx::id_type>& remoteIDs) :
        messageNeighborIDs(messageNeighborIDs),
        messagesFromNeighbors(hpx::util::unwrapped(messagesFromNeighbors)),
        remoteIDs(remoteIDs)
    {}

    const MESSAGE& operator[](int index) const
    {
        std::vector<int>::const_iterator i = std::find(messageNeighborIDs.begin(), messageNeighborIDs.end(), index);
        if (i == messageNeighborIDs.end()) {
            throw std::logic_error("ID not found for incoming messages");
        }

        return messagesFromNeighbors[i - messageNeighborIDs.begin()];
    }

    void send(int remoteCellID, const MESSAGE& message, int step) const
    {
        std::map<int, hpx::id_type>::const_iterator iter = remoteIDs.find(remoteCellID);
        if (iter == remoteIDs.end()) {
            throw std::logic_error("ID not found for outgoing messages");
        }

        hpx::apply(
            typename HPXReceiver<MESSAGE>::receiveAction(),
            iter->second,
            step,
            message);
    }

private:
    std::vector<int> messageNeighborIDs;
    std::vector<MESSAGE> messagesFromNeighbors;
    std::map<int, hpx::id_type> remoteIDs;
};

// fixme: componentize
template<typename CELL, typename MESSAGE>
class CellComponent
{
public:
    typedef DisplacedGrid<CELL, Topologies::Unstructured::Topology> GridType;

    // fixme: move semantics
    explicit CellComponent(
        boost::shared_ptr<GridType> grid = 0,
        int id = -1,
        std::vector<int> neighbors = std::vector<int>()) :
        grid(grid),
        id(id)
    {
        for (auto&& neighbor: neighbors) {
            std::string linkName = endpointName(neighbor, id);
            receivers[neighbor] = HPXReceiver<MESSAGE>::make(linkName).get();
        }
    }

    // fixme: use move semantics here
    void update(
        std::vector<int> neighbors,
        std::vector<hpx::shared_future<MESSAGE> > inputFutures,
        const hpx::shared_future<void>& /* unused, just here to ensure
                                           correct ordering of updates
                                           per cell */,
        int step)
    {
        Neighborhood<MESSAGE> hood(neighbors, inputFutures, remoteIDs);
        cell()->update(hood, step + 1);
    }

    static std::string endpointName(int sender, int receiver)
    {
        // fixme: make this prefix configurable
        return "HPXDataflowSimulatorEndPoint_" +
            StringOps::itoa(sender) +
            "_to_" +
            StringOps::itoa(receiver);

    }

    boost::shared_ptr<GridType> grid;
    int id;
    std::map<int, std::shared_ptr<HPXReceiver<MESSAGE> > > receivers;
    std::map<int, hpx::id_type> remoteIDs;

    CELL *cell()
    {
        return grid->baseAddress();
    }
};

}

/**
 * Experimental Simulator based on (surprise surprise) HPX' dataflow
 operator. Primary use case (for now) is DGSWEM.
 */
template<typename CELL, typename PARTITION = UnstructuredStripingPartition>
class HPXDataflowSimulator : public HierarchicalSimulator<CELL>
{
public:
    typedef typename APITraits::SelectMessageType<CELL>::Value MessageType;
    typedef HierarchicalSimulator<CELL> ParentType;
    typedef typename DistributedSimulator<CELL>::Topology Topology;
    typedef PartitionManager<Topology> PartitionManagerType;
    using DistributedSimulator<CELL>::NANO_STEPS;
    using DistributedSimulator<CELL>::initializer;
    using HierarchicalSimulator<CELL>::initialWeights;

    inline HPXDataflowSimulator(
        Initializer<CELL> *initializer,
        int loadBalancingPeriod = 10000,
        bool enableFineGrainedParallelism = true) :
        ParentType(
            initializer,
            loadBalancingPeriod * NANO_STEPS,
            enableFineGrainedParallelism)
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
        // move to simulator:
        typedef hpx::shared_future<void> UpdateResultFuture;
        typedef std::vector<UpdateResultFuture> TimeStepFutures;

        using hpx::dataflow;
        using hpx::util::unwrapped;
        TimeStepFutures lastTimeStepFutures;
        TimeStepFutures thisTimeStepFutures;

        Region<1> localRegion;
        CoordBox<1> box = initializer->gridBox();
        std::size_t rank = hpx::get_locality_id();
        std::size_t numLocalities = hpx::get_num_localities().get();

        std::vector<double> rankSpeeds(numLocalities, 1.0);
        std::vector<std::size_t> weights = initialWeights(
            box.dimensions.prod(),
            rankSpeeds);

        Region<1> globalRegion;
        globalRegion << box;

        boost::shared_ptr<PARTITION> partition(
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
        boost::shared_ptr<Adjacency> adjacency = initializer->getAdjacency(localRegion);

        // fixme: instantiate components in agas and only hold ids of those
        typedef HPXDataFlowSimulatorHelpers::CellComponent<CELL, MessageType> ComponentType;
        typedef typename ComponentType::GridType GridType;

        std::map<int, ComponentType> components;
        std::vector<int> neighbors;

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            int id = i->x();
            CoordBox<1> singleCellBox(Coord<1>(id), Coord<1>(1));
            boost::shared_ptr<GridType> grid(new GridType(singleCellBox));
            initializer->grid(&*grid);

            neighbors.clear();
            adjacency->getNeighbors(id, &neighbors);

            ComponentType component(grid, id, neighbors);
            components[id] = component;
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            ComponentType& component = components[i->x()];

            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);

            // fixme: move this initialization into the c-tor of the CellComponent:
            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                std::string linkName = HPXDataFlowSimulatorHelpers::CellComponent<MessageType, MessageType>::endpointName(
                    i->x(), *j);
                component.remoteIDs[*j] = hpx::id_type(HPXReceiver<MessageType>::find(linkName).get());
            }
        }

        // fixme: also create dataflow in cellcomponent
        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            lastTimeStepFutures << hpx::make_ready_future(UpdateResultFuture());
        }
        thisTimeStepFutures.resize(localRegion.size());

	// HPX Reset counters 
	hpx::reset_active_counters();

        // fixme: add steerer/writer interaction
        int maxTimeSteps = initializer->maxSteps();
        for (int t = 0; t < maxTimeSteps; ++t) {
            int index = 0;

            for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {

                std::vector<hpx::shared_future<MessageType> > receiveMessagesFutures;
                neighbors.clear();
                adjacency->getNeighbors(i->x(), &neighbors);

                    for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                        if (t > 0) {
                            receiveMessagesFutures << components[i->x()].receivers[*j]->get(t);
                        } else {
                            int data = *j * 100 + i->x();
                            receiveMessagesFutures <<  hpx::make_ready_future(MessageType());
                        }
                }

                auto Operation = boost::bind(&ComponentType::update,
                                             components[i->x()], _1, _2, _3, _4);

                thisTimeStepFutures[index] = dataflow(
                    hpx::launch::async,
                    Operation,
                    neighbors,
                    receiveMessagesFutures,
                    lastTimeStepFutures[index],
                    // fixme: nanoStep!
                    t);

                ++index;
            }

            using std::swap;
            swap(thisTimeStepFutures, lastTimeStepFutures);
        }

        hpx::when_all(lastTimeStepFutures).get();
    }

    std::vector<Chronometer> gatherStatistics()
    {
        // fixme
        return std::vector<Chronometer>();
    }

private:

};

}

#endif

#endif
