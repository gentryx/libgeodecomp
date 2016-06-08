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
    DummyMessage(int senderID = -1,
                 int receiverID = -1,
                 int timestep = -1,
                 int data = -1) :
        senderID(senderID),
        receiverID(receiverID),
        timestep(timestep),
        data(data)
    {}

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int)
    {
        archive & senderID;
        archive & receiverID;
        archive & timestep;
        archive & data;
    }

    int senderID;
    int receiverID;
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
        // fixme: make sure nanosteps are being issued here, not global steps:
        int nanoStep)
    {
	for (auto&& neighbor: neighbors) {
            // fixme: use actual step AND nanoStep here
            int expectedData = 10000 * (nanoStep - 1) + neighbor * 100 + id;
            TS_ASSERT_EQUALS(hood[neighbor].data,       expectedData);
            TS_ASSERT_EQUALS(hood[neighbor].timestep,   (nanoStep - 1));
            TS_ASSERT_EQUALS(hood[neighbor].senderID,   neighbor);
            TS_ASSERT_EQUALS(hood[neighbor].receiverID, id);
	}

        for (auto&& neighbor: neighbors) {
            // fixme: use actual step AND nanoStep here
            DummyMessage dummyMessage(id, neighbor, nanoStep, 10000 * nanoStep + 100 * id + neighbor);
            // fixme: strip this from signature
            hood.send(neighbor, dummyMessage, nanoStep);
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

         if (id > 0) {
             neighbors << (id - 1);
         }
         if (id > 1) {
             neighbors << (id - 2);
         }

         if (id < (gridSize - 1)) {
             neighbors << (id + 1);
         }
         if (id < (gridSize - 2)) {
             neighbors << (id + 2);
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
    // fixme: move semantics
    explicit TestComponent(CELL *cell = 0, int id = -1, std::vector<int> neighbors = std::vector<int>()) :
        cell(cell),
        id(id)
    {
        for (auto&& neighbor: neighbors) {
            std::string linkName = TestComponent<DummyMessage>::endpointName(neighbor, id);
            receivers[neighbor] = HPXReceiver<DummyMessage>::make(linkName).get();
        }
    }

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

    static std::string endpointName(int sender, int receiver)
    {
        return "HPXDataflowSimulatorEndPoint_" +
            StringOps::itoa(sender) +
            "_to_" +
            StringOps::itoa(receiver);

    }

    CELL *cell;
    int id;
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
        typedef std::vector<UpdateResultFuture> TimeStepFutures;

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
        std::vector<int> neighbors;

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);
            TestComponent<DummyModel> component(&grid[*i], i->x(), neighbors);
            components[i->x()] = component;
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            TestComponent<DummyModel>& component = components[i->x()];

            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);

            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                std::string linkName = TestComponent<DummyMessage>::endpointName(i->x(), *j);
                component.remoteIDs[*j] = hpx::id_type(HPXReceiver<DummyMessage>::find(linkName).get());
            }
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            lastTimeStepFutures << hpx::make_ready_future(UpdateResultFuture());
        }
        thisTimeStepFutures.resize(localRegion.size());

        int maxTimeSteps = initializer.maxSteps();
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

                auto Operation = boost::bind(&TestComponent<DummyModel>::update, components[i->x()], _1, _2, _3, _4);

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
};

}
