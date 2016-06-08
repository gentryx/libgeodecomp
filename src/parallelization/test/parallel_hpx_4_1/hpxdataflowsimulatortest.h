#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/parallelization/hpxdataflowsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DummyMessage
{
public:
    DummyMessage(int senderId = -1,
                 int receiverId = -1,
                 int timestep = -1,
                 int data = -1) :
        senderId(senderId),
        receiverId(receiverId),
        timestep(timestep),
        data(data)
    {}

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int)
    {
        archive & senderId;
        archive & receiverId;
        archive & timestep;
        archive & data;
    }

    int senderId;
    int receiverId;
    int timestep;
    int data;
};

}

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(DummyMessage)

namespace LibGeoDecomp {

class DummyModel
{
public:
    class API :
        public APITraits::HasUnstructuredTopology
    {};

    DummyModel(int id = -1, const std::vector<int>& neighbors = std::vector<int>()) :
        id(id),
        neighbors(neighbors)
    {}

    // fixme: use move semantics here
    template<typename HOOD>
    void update(
        HOOD& hood,
        int step)
    {
	std::cout << "updating Dummy " << id << " and my neighbors are: [\n";
	for (int i = 0; i != neighbors.size(); ++i) {
	    std::cout << "  at " << neighbors[i]
                      << ", " << hood[neighbors[i]].senderId
                      << " -> " << hood[neighbors[i]].receiverId << "\n";
	}
	std::cout << "]\n";

        for (auto&& neighbor: neighbors) {
            DummyMessage dummyMessage(id, neighbor, step, 10000 * step + 100 * id + neighbor);
            // fixme: strip this from signature
            hood.send(neighbor, dummyMessage, step);
        }
    }

private:
    int id;
    std::vector<int> neighbors;
};

 class DummyInitializer : public Initializer<DummyModel>
 {
 public:
     DummyInitializer(int gridSize, int myMaxSteps) :
         gridSize(gridSize),
	 myMaxSteps(myMaxSteps)
     {}

     void grid(GridBase<DummyModel, 1> *grid)
     {
	 CoordBox<1> box = grid->boundingBox();

	 for (CoordBox<1>::Iterator i = box.begin(); i != box.end(); ++i) {
             DummyModel cell(i->x(), getNeighbors(i->x()));
             std::cout << "initing id " << i->x() << " with neighbors " << getNeighbors(i->x()) << "\n";
	     grid->set(*i, cell);
	 }
     }

    virtual Coord<1> gridDimensions() const
    {
	return Coord<1>(gridSize);
    }

    unsigned startStep() const
    {
	return 0;
    }

    unsigned maxSteps() const
    {
	return myMaxSteps;
    }

    boost::shared_ptr<Adjacency> getAdjacency(const Region<1>& region) const
    {
	boost::shared_ptr<Adjacency> adjacency(new RegionBasedAdjacency());

	for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            std::vector<int> neighbors = getNeighbors(i->x());
            for (auto&& neighbor: neighbors) {
                adjacency->insert(i->x(), neighbor);
            }
	}

	return adjacency;
    }

 private:
     int gridSize;
     int myMaxSteps;

     std::vector<int> getNeighbors(int id) const
     {
         std::vector<int> neighbors;
         // fixme: have more connections
         if (id != 0) {
             neighbors << (id - 1);
         }

         if (id != (gridSize - 1)) {
             neighbors << (id + 1);
         }

         return neighbors;
     }
};

template<typename MESSAGE>
class HPXDataflowNeighborhood
{
public:
    // fixme: move semantics
    inline HPXDataflowNeighborhood(
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

    void send(int remoteCellID, const DummyMessage& message, int step) const
    {
        std::map<int, hpx::id_type>::const_iterator iter = remoteIDs.find(remoteCellID);
        if (iter == remoteIDs.end()) {
            throw std::logic_error("ID not found for outgoing messages");
        }

        hpx::apply(
            typename HPXReceiver<DummyMessage>::receiveAction(),
            iter->second,
            step,
            message);
    }

private:
    std::vector<int> messageNeighborIDs;
    std::vector<MESSAGE> messagesFromNeighbors;
    std::map<int, hpx::id_type> remoteIDs;
};
template<typename CELL>
class TestComponent
{
public:
    explicit TestComponent(CELL *cell = 0) :
        cell(cell)
    {}

    // fixme: use move semantics here
    void update(
        std::vector<int> neighbors,
        std::vector<hpx::shared_future<DummyMessage> > inputFutures,
        const hpx::shared_future<void>& /* unused, just here to ensure
                                           correct ordering of updates
                                           per cell */,
        int step)
    {
        HPXDataflowNeighborhood<DummyMessage> hood(neighbors, inputFutures, remoteIDs);
        cell->update(hood, step + 1);
    }

    CELL *cell;
    std::map<int, std::shared_ptr<HPXReceiver<DummyMessage> > > receivers;
    std::map<int, hpx::id_type> remoteIDs;
};

class HpxDataflowSimulatorTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
    }

    void testBasic()
    {
        DummyInitializer initializer(50, 13);
        UnstructuredGrid<DummyModel> grid(initializer.gridBox());
        initializer.grid(&grid);

        typedef hpx::shared_future<void> UpdateResultFuture;
        typedef std::map<int, UpdateResultFuture> TimeStepFutures;

        using hpx::dataflow;
        using hpx::util::unwrapped;
        TimeStepFutures lastTimeStepFutures;
        TimeStepFutures thisTimeStepFutures;

        Region<1> localRegion;
        CoordBox<1> box = initializer.gridBox();
        int rank = hpx::get_locality_id();
        for (int i = ((rank + 0) * box.dimensions.x() / 4);
             i <     ((rank + 1) * box.dimensions.x() / 4);
             ++i) {
            localRegion << Coord<1>(i);
        }

        boost::shared_ptr<Adjacency> adjacency = initializer.getAdjacency(localRegion);

        // fixme: instantiate components in agas and only hold ids of those
        std::map<int, TestComponent<DummyModel> > components;
        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            TestComponent<DummyModel> component(&grid[*i]);

            std::vector<int> neighbors;
            adjacency->getNeighbors(i->x(), &neighbors);

            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                component.receivers[*j] = HPXReceiver<DummyMessage>::make(
                    "hpx_receiver_" +
                    StringOps::itoa(*j) +
                    "_to_" +
                    StringOps::itoa(i->x())).get();
            }

            components[i->x()] = component;
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            TestComponent<DummyModel>& component = components[i->x()];

            std::vector<int> neighbors;
            adjacency->getNeighbors(i->x(), &neighbors);

            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                std::string linkName = "hpx_receiver_" +
                    StringOps::itoa(i->x()) +
                    "_to_" +
                    StringOps::itoa(*j);

                component.remoteIDs[*j] = hpx::id_type(HPXReceiver<DummyMessage>::find(linkName).get());
            }
        }

        std::cout << "setting up dataflow\n";
        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            lastTimeStepFutures[i->x()] = hpx::make_ready_future(UpdateResultFuture());
        }

        int maxTimeSteps = initializer.maxSteps();
        for (int t = 0; t < maxTimeSteps; ++t) {

            for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {

                std::vector<hpx::shared_future<DummyMessage> > receiveMessagesFutures;
                std::vector<int> neighbors;
                adjacency->getNeighbors(i->x(), &neighbors);

                    for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                        if (t > 0) {
                            receiveMessagesFutures << components[i->x()].receivers[*j]->get(t);
                        } else {
                            receiveMessagesFutures <<  hpx::make_ready_future(DummyMessage(-1, -1, -1, -1));
                        }
                }

                auto Operation = boost::bind(&TestComponent<DummyModel>::update, components[i->x()], _1, _2, _3, _4);

                thisTimeStepFutures[i->x()] = dataflow(
                    hpx::launch::async,
                    Operation,
                    neighbors,
                    receiveMessagesFutures,
                    lastTimeStepFutures[i->x()],
                    t);
            }

            using std::swap;
            swap(thisTimeStepFutures, lastTimeStepFutures);
        }

        // fixme: this is ugly
        for (auto&& i: lastTimeStepFutures) {
            i.second.get();
        }
        // hpx::when_all(lastTimeStepFutures).get();

        std::cout << "dataflow test done\n";
    }
};

}
