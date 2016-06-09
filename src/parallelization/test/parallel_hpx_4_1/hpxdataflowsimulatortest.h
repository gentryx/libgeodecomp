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

        // move to simulator:
        typedef hpx::shared_future<void> UpdateResultFuture;
        typedef std::vector<UpdateResultFuture> TimeStepFutures;

        using hpx::dataflow;
        using hpx::util::unwrapped;
        TimeStepFutures lastTimeStepFutures;
        TimeStepFutures thisTimeStepFutures;

        Region<1> localRegion;
        CoordBox<1> box = initializer.gridBox();
        int rank = hpx::get_locality_id();
        // fixme: don't hardcode num of localities
        for (int i = ((rank + 0) * box.dimensions.x() / 4);
             i <     ((rank + 1) * box.dimensions.x() / 4);
             ++i) {
            localRegion << Coord<1>(i);
        }

        boost::shared_ptr<Adjacency> adjacency = initializer.getAdjacency(localRegion);

        // fixme: instantiate components in agas and only hold ids of those
        std::map<int, HPXDataFlowSimulatorHelpers::CellComponent<DummyModel, DummyMessage> > components;
        std::vector<int> neighbors;

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            neighbors.clear();
            adjacency->getNeighbors(i->x(), &neighbors);
            HPXDataFlowSimulatorHelpers::CellComponent<DummyModel, DummyMessage> component(&grid[*i], i->x(), neighbors);
            components[i->x()] = component;
        }

        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            HPXDataFlowSimulatorHelpers::CellComponent<DummyModel, DummyMessage>& component = components[i->x()];

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

                auto Operation = boost::bind(&HPXDataFlowSimulatorHelpers::CellComponent<DummyModel, DummyMessage>::update,
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
};

}
