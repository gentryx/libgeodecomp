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
        id(id)
    {}

    // fixme: use move semantics here
    template<typename HOOD>
    void update(
        std::vector<hpx::shared_future<DummyMessage> > inputFutures,
        int step,
        const HOOD& hood)
    {
	std::cout << "updating Dummy " << id << " and my neighbors are: [";
        std::vector<DummyMessage> input = hpx::util::unwrapped(inputFutures);

	for (int i = 0; i != input.size(); ++i) {
	    std::cout << input[i].senderId << " -> " << input[i].receiverId;
	}
	std::cout << "]\n";

        for (auto&& neighbor: neighbors) {
            DummyMessage dummyMessage;
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

template<typename CELL>
class TestComponent
{
public:
    explicit TestComponent(CELL *cell = 0) :
        cell(cell)
    {}

    // fixme: use move semantics here
    // fixme: int -> void?
    int update(
        std::vector<hpx::shared_future<DummyMessage> > inputFutures,
        const hpx::shared_future<int>& /* unused */,
        int step)
    {
        cell->update(inputFutures, step, *this);
        return 0;
    }

    void send(int remoteCellID, const DummyMessage& message, int step) const
    {
        std::map<int, hpx::id_type>& nonConstRemoteIDs = const_cast<std::map<int, hpx::id_type>&>(remoteIDs);
        hpx::apply(
            typename HPXReceiver<DummyMessage>::receiveAction(),
            nonConstRemoteIDs[remoteCellID],
            step,
            message);
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
        std::cout << "starting dataflow test\n";

        DummyInitializer initializer(50, 13);
        UnstructuredGrid<DummyModel> grid(initializer.gridBox());
        initializer.grid(&grid);
        std::cout << "grid size: " << grid.boundingBox() << "\n";

        typedef hpx::shared_future<int> UpdateResultFuture;
        typedef std::map<int, UpdateResultFuture> TimeStepFutures;

        using hpx::dataflow;
        using hpx::util::unwrapped;
        TimeStepFutures lastTimeStepFutures;
        TimeStepFutures thisTimeStepFutures;
        std::cout << "blah0\n";

        Region<1> localRegion;
        CoordBox<1> box = initializer.gridBox();
        int rank = hpx::get_locality_id();
        for (int i = ((rank + 0) * box.dimensions.x() / 4);
             i <     ((rank + 1) * box.dimensions.x() / 4);
             ++i) {
            localRegion << Coord<1>(i);
        }

        boost::shared_ptr<Adjacency> adjacency = initializer.getAdjacency(localRegion);
        std::cout << "blah1\n";

        // fixme: instantiate components in agas and only hold ids of those
        std::map<int, TestComponent<DummyModel> > components;
        for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
            std::cout << "   blubbA " << *i << "\n";
            TestComponent<DummyModel> component(&grid[*i]);

            std::cout << "   blubbB " << *i << "\n";
            std::vector<int> neighbors;
            adjacency->getNeighbors(i->x(), &neighbors);
            std::cout << "   blubbC " << neighbors << "\n";

            for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                component.receivers[*j] = HPXReceiver<DummyMessage>::make(
                    "hpx_receiver_" +
                    StringOps::itoa(*j) +
                    "_to_" +
                    StringOps::itoa(i->x())).get();
            }

            std::cout << "   blubbD " << *i << "\n";
            components[i->x()] = component;
            std::cout << "   blubbE " << *i << "\n";
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

        std::cout << "blah2\n";
        int maxTimeSteps = initializer.maxSteps();
        for (int t = 0; t < maxTimeSteps; ++t) {

            for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {

                std::vector<hpx::shared_future<DummyMessage> > receiveMessagesFutures;
                std::vector<int> neighbors;
                adjacency->getNeighbors(i->x(), &neighbors);
                for (auto j = neighbors.begin(); j != neighbors.end(); ++j) {
                    receiveMessagesFutures << hpx::make_ready_future(DummyMessage(-1, -1, -1, -1));
                    // receiveMessagesFutures << components[i->x()].receivers[*j].get(t);
                }

                auto Operation = boost::bind(&TestComponent<DummyModel>::update, components[i->x()], _1, _2, _3);

                thisTimeStepFutures[i->x()] = dataflow(
                    hpx::launch::async,
                    Operation,
                    receiveMessagesFutures,
                    lastTimeStepFutures[i->x()],
                    t);

                // for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {

                //     // fixme: use hpxreceiver::receive to get futures
                //     auto Op = unwrapped(boost::bind(&DummyModel::update, &grid[i->x()], _1));
                //     std::vector<ResultsFuture> localDependencies;

                //     std::vector<int> neighbors;
                //     adjacency->getNeighbors(i->x(), &neighbors);

                //     for (std::vector<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n) {
                //         localDependencies.push_back(lastTimeStepFutures[*n]);
                //     }

                //     thisTimeStepFutures[i->x()] = dataflow(hpx::launch::async, Op, localDependencies);
            }

            using std::swap;
            swap(thisTimeStepFutures, lastTimeStepFutures);
        }

        std::cout << "waiting on futures\n";

        // std::vector<ResultsFuture> finalStep;
        // for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
        //     finalStep.push_back(futures[maxTimeSteps % 2][i->x()]);
        // }

        // fixme: this is ugly
        for (auto&& i: lastTimeStepFutures) {
            i.second.get();
        }
        // hpx::when_all(lastTimeStepFutures).get();

        std::cout << "dataflow test done\n";
    }
};

}
