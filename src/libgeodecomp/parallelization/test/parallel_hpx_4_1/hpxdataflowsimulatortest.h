#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>

#include <libgeodecomp/geometry/partitions/ptscotchunstructuredpartition.h>
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
    static const int NANO_STEPS = 3;

    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasNanoSteps<NANO_STEPS>,
        public APITraits::HasCustomMessageType<DummyMessage>
    {};

    DummyModel(int id = -1, const std::vector<int>& neighbors = std::vector<int>()) :
        id(id),
        neighbors(neighbors)
    {}

    template<typename HOOD, typename EVENT>
    void update(
        HOOD&& hood,
        const EVENT& event)
    {
        int globalNanoStep = event.step() * NANO_STEPS + event.nanoStep();

        std::vector<int> sortedNeighbors = neighbors;
        std::sort(sortedNeighbors.begin(), sortedNeighbors.end());
        TS_ASSERT_EQUALS(hood.neighbors(), sortedNeighbors);

        if ((globalNanoStep) > 0) {
            for (auto&& neighbor: neighbors) {
                int expectedData = 10000 * globalNanoStep + neighbor * 100 + id;
                TS_ASSERT_EQUALS(hood[neighbor].data,       expectedData);
                TS_ASSERT_EQUALS(hood[neighbor].timestep,   globalNanoStep);
                TS_ASSERT_EQUALS(hood[neighbor].senderID,   neighbor);
                TS_ASSERT_EQUALS(hood[neighbor].receiverID, id);
            }
        }

        for (auto&& neighbor: neighbors) {
            DummyMessage dummyMessage(id, neighbor, globalNanoStep + 1, 10000 * (globalNanoStep + 1) + 100 * id + neighbor);
            hood.send(neighbor, dummyMessage);
        }
    }

private:
    int id;
    std::vector<int> neighbors;
};

class AsymmetricDummyModel
{
public:
    static const int NANO_STEPS = 3;

    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasNanoSteps<NANO_STEPS>,
        public APITraits::HasCustomMessageType<DummyMessage>
    {};

    AsymmetricDummyModel(int id = -1, const std::vector<int>& neighbors = std::vector<int>()) :
        id(id),
        neighbors(neighbors)
    {}

    template<typename HOOD, typename EVENT>
    void update(
        HOOD&& hood,
        const EVENT& event)
    {
        int globalNanoStep = event.step() * NANO_STEPS + event.nanoStep();

        if ((globalNanoStep) > 0) {
            for (auto&& neighbor: neighbors) {
                if (neighbor < id) {
                    int expectedData = 10000 * globalNanoStep + neighbor * 100 + id;
                    TS_ASSERT_EQUALS(hood[neighbor].data,       expectedData);
                    TS_ASSERT_EQUALS(hood[neighbor].timestep,   globalNanoStep);
                    TS_ASSERT_EQUALS(hood[neighbor].senderID,   neighbor);
                    TS_ASSERT_EQUALS(hood[neighbor].receiverID, id);
                }
            }
        }

        for (auto&& neighbor: neighbors) {
            if (neighbor > id) {
                DummyMessage dummyMessage(id, neighbor, globalNanoStep + 1, 10000 * (globalNanoStep + 1) + 100 * id + neighbor);
                hood.send(neighbor, dummyMessage);
            }
        }
    }

private:
    int id;
    std::vector<int> neighbors;
};

}

REGISTER_CELLCOMPONENT(DummyModel, DummyMessage, fixmeB)
REGISTER_CELLCOMPONENT(AsymmetricDummyModel, DummyMessage, fixmeC)

namespace LibGeoDecomp {


template<typename MODEL>
class DummyInitializer : public Initializer<MODEL>
{
public:
    using typename Initializer<MODEL>::AdjacencyPtr;

    DummyInitializer(int gridSize, int myMaxSteps) :
        gridSize(gridSize),
        myMaxSteps(myMaxSteps)
    {}

    void grid(GridBase<MODEL, 1> *grid)
    {
        CoordBox<1> box = grid->boundingBox();
        typename GridBase<MODEL, 1>::SparseMatrix weights;

        for (CoordBox<1>::Iterator i = box.begin(); i != box.end(); ++i) {
            std::vector<int> neighbors = getNeighbors(i->x());
            MODEL cell(i->x(), neighbors);
            grid->set(*i, cell);

            for (auto&& j: neighbors) {
                weights << std::make_pair(Coord<2>(i->x(), j), 0.1);
            }
        }

        grid->setWeights(0, weights);
    }

    Coord<1> gridDimensions() const
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

    AdjacencyPtr getAdjacency(const Region<1>& region) const
    {
	AdjacencyPtr adjacency(new RegionBasedAdjacency());

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
        Initializer<DummyModel> *initializer = new DummyInitializer<DummyModel>(50, 13);
        HPXDataflowSimulator<DummyModel> sim(initializer, "testBasic");
        sim.run();
    }

    // intentionally commented out as this test would run
    // prohibitively long (about 6 min).

    // void testManySteps()
    // {
    //     Chronometer c;
    //     int steps = 300000;
    //     {
    //         TimeTotal timer(&c);
    //         Initializer<DummyModel> *initializer = new DummyInitializer<DummyModel>(60, steps);
    //         HPXDataflowSimulator<DummyModel> sim(initializer, "testManySteps");
    //         sim.run();
    //     }
    //     std::cout << "steps: " << steps << ", time: " << c.interval<TimeTotal>() << "\n";
    // }

    void testAsymmetric()
    {
        Initializer<AsymmetricDummyModel> *initializer = new DummyInitializer<AsymmetricDummyModel>(50, 13);
        HPXDataflowSimulator<AsymmetricDummyModel> sim(initializer, "testAsymmetric");
        sim.run();
    }

    void testPTScotch()
    {
#ifdef LIBGEODECOMP_WITH_SCOTCH
        Initializer<DummyModel> *initializer = new DummyInitializer<DummyModel>(50, 13);
        HPXDataflowSimulator<DummyModel, PTScotchUnstructuredPartition<1> > sim(initializer, "testPTScotch");
        sim.run();
#endif
    }

};

}
