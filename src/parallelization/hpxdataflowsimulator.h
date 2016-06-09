#ifndef LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>

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
    // fixme: move semantics
    explicit CellComponent(CELL *cell = 0, int id = -1, std::vector<int> neighbors = std::vector<int>()) :
        cell(cell),
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
        cell->update(hood, step + 1);
    }

    static std::string endpointName(int sender, int receiver)
    {
        return "HPXDataflowSimulatorEndPoint_" +
            StringOps::itoa(sender) +
            "_to_" +
            StringOps::itoa(receiver);

    }

    CELL *cell;
    int id;
    std::map<int, std::shared_ptr<HPXReceiver<MESSAGE> > > receivers;
    std::map<int, hpx::id_type> remoteIDs;
};

}

// fixme
class DummyMessage;

/**
 * Experimental Simulator based on (surprise surprise) HPX' dataflow
 operator. Primary use case (for now) is DGSWEM.
 */
template<typename CELL>
class HPXDataflowSimulator : public DistributedSimulator<CELL>
{
public:
    using DistributedSimulator<CELL>::initializer;

    inline HPXDataflowSimulator(Initializer<CELL> *initializer) :
        DistributedSimulator<CELL>(initializer)
    {}

    void step()
    {
        // fixme
    }

    void run()
    {
        UnstructuredGrid<CELL> grid(initializer->gridBox());
        initializer->grid(&grid);
        // move to simulator:
        typedef hpx::shared_future<void> UpdateResultFuture;
        typedef std::vector<UpdateResultFuture> TimeStepFutures;

        using hpx::dataflow;
        using hpx::util::unwrapped;
        TimeStepFutures lastTimeStepFutures;
        TimeStepFutures thisTimeStepFutures;

        Region<1> localRegion;
        CoordBox<1> box = initializer->gridBox();
        int rank = hpx::get_locality_id();
        // fixme: don't hardcode num of localities
        for (int i = ((rank + 0) * box.dimensions.x() / 4);
             i <     ((rank + 1) * box.dimensions.x() / 4);
             ++i) {
            localRegion << Coord<1>(i);
        }

        boost::shared_ptr<Adjacency> adjacency = initializer->getAdjacency(localRegion);

        // fixme: instantiate components in agas and only hold ids of those
        std::map<int, HPXDataFlowSimulatorHelpers::CellComponent<CELL, DummyMessage> > components;
        std::vector<int> neighbors;

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);
            HPXDataFlowSimulatorHelpers::CellComponent<CELL, DummyMessage> component(&grid[*i], i->x(), neighbors);
            components[i->x()] = component;
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            HPXDataFlowSimulatorHelpers::CellComponent<CELL, DummyMessage>& component = components[i->x()];

            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);

            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                std::string linkName = HPXDataFlowSimulatorHelpers::CellComponent<DummyMessage, DummyMessage>::endpointName(
                    i->x(), *j);
                component.remoteIDs[*j] = hpx::id_type(HPXReceiver<DummyMessage>::find(linkName).get());
            }
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            lastTimeStepFutures << hpx::make_ready_future(UpdateResultFuture());
        }
        thisTimeStepFutures.resize(localRegion.size());

        // fixme: add steerer/writer interaction
        int maxTimeSteps = initializer->maxSteps();
        for (int t = 0; t < maxTimeSteps; ++t) {
            int index = 0;

            for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {

                std::vector<hpx::shared_future<DummyMessage> > receiveMessagesFutures;
                neighbors.clear();
                adjacency->getNeighbors(i->x(), &neighbors);

                    for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                        if (t > 0) {
                            receiveMessagesFutures << components[i->x()].receivers[*j]->get(t);
                        } else {
                            int data = *j * 100 + i->x();
                            receiveMessagesFutures <<  hpx::make_ready_future(DummyMessage(*j, i->x(), 0, data));
                        }
                }

                auto Operation = boost::bind(&HPXDataFlowSimulatorHelpers::CellComponent<CELL, DummyMessage>::update,
                                             components[i->x()], _1, _2, _3, _4);

                thisTimeStepFutures[index] = dataflow(
                    hpx::launch::async,
                    Operation,
                    neighbors,
                    receiveMessagesFutures,
                    lastTimeStepFutures[index],
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
