#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <boost/assign/std/deque.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;
using namespace boost::assign;

class ConwayCell
{
public:
    ConwayCell(bool alive = false) :
        alive(alive)
    {}

    int countLivingNeighbors(const CoordMap<ConwayCell>& neighborhood)
    {
        int ret = 0;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                ret += neighborhood[Coord<2>(x, y)].alive;
            }
        }
        ret -= neighborhood[Coord<2>(0, 0)].alive;
        return ret;
    }

    void update(const CoordMap<ConwayCell>& neighborhood, const unsigned&)
    {
        int livingNeighbors = countLivingNeighbors(neighborhood);
        alive = neighborhood[Coord<2>(0, 0)].alive;
        if (alive) {
            alive = (2 <= livingNeighbors) && (livingNeighbors <= 3);
        } else {
            alive = (livingNeighbors == 3);
        }
    }

    bool alive;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & alive;
    }
};

class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    CellInitializer(int maxTimeSteps = 100) :
        SimpleInitializer<ConwayCell>(Coord<2>(160, 90), maxTimeSteps)
    {}

    virtual void grid(GridBase<ConwayCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        std::vector<Coord<2> > startCells;
        // start with a single glider...
        //          x
        //           x
        //         xxx
        startCells +=
            Coord<2>(11, 10),
            Coord<2>(12, 11),
            Coord<2>(10, 12), Coord<2>(11, 12), Coord<2>(12, 12);


        // ...add a Diehard pattern...
        //                x
        //          xx
        //           x   xxx
        startCells +=
            Coord<2>(55, 70), Coord<2>(56, 70), Coord<2>(56, 71),
            Coord<2>(60, 71), Coord<2>(61, 71), Coord<2>(62, 71),
            Coord<2>(61, 69);

        // ...and an Acorn pattern:
        //        x
        //          x
        //       xx  xxx
        startCells +=
            Coord<2>(111, 30),
            Coord<2>(113, 31),
            Coord<2>(110, 32), Coord<2>(111, 32),
            Coord<2>(113, 32), Coord<2>(114, 32), Coord<2>(115, 32);


        for (std::vector<Coord<2> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i) {
            if (rect.inBounds(*i)) {
                ret->set(*i, ConwayCell(true));
            }
        }
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<SimpleInitializer<ConwayCell> >(*this);
    }
};

typedef
    HpxSimulator::HpxSimulator<ConwayCell, RecursiveBisectionPartition<2> >
    SimulatorType;

typedef LibGeoDecomp::SerialBOVWriter<ConwayCell> BovWriterType;

typedef
    LibGeoDecomp::HpxWriterCollector<ConwayCell>
    HpxWriterCollectorType;

namespace LibGeoDecomp {

class HpxSimulatorTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        outputFrequency = 1;
        maxTimeSteps = 10;
        rank = hpx::get_locality_id();
        localities = hpx::find_all_localities();
    }

    void tearDown()
    {
        int i;
        for (i = 0; i < maxTimeSteps; i += outputFrequency) {
            removeFiles(i);
        }

        if (i > maxTimeSteps) {
            i = maxTimeSteps;
        }
        removeFiles(i);
    }

    void testBasic()
    {
        // works:
        CellInitializer *init = new CellInitializer(maxTimeSteps);

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            "/0_fixme_HPXSimulatorUpdateGroupSdfafafasdasd");

        // fixme: add writer!
        // HpxWriterCollectorType::SinkType sink(
        //     new BovWriterType(&ConwayCell::alive, "game", outputFrequency),
        //     sim.numUpdateGroups());

        // sim.addWriter(new HpxWriterCollectorType(sink));

        sim.run();

        // if (rank == 0) {
        //     hpx::lcos::broadcast<barrier_action>(localities, std::string("testBasic"));
        // }
        // p["testBasic"].get_future().get();
    }

    void fastestHeterogeneous()
    {
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        CellInitializer *init = new CellInitializer(maxTimeSteps);

        // std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        // int loadBalancingPeriod = 10;
        // int ghostZoneWidth = 1;
        // SimulatorType sim(
        //     init,
        //     updateGroupSpeeds,
        //     new TracingBalancer(new OozeBalancer()),
        //     loadBalancingPeriod,
        //     ghostZoneWidth,
        //     "HpxSimulatorTestTestHeterogeneous");

        // // fixme: add writer!
        // // HpxWriterCollectorType::SinkType sink(
        // //     new BovWriterType(&ConwayCell::alive, "game", outputFrequency),
        // //     sim.numUpdateGroups());

        // // sim.addWriter(new HpxWriterCollectorType(sink));

        // std::cout << "testC\n";
        // sim.run();
        // std::cout << "testD\n";

        // if (rank == 0) {
        //     hpx::lcos::broadcast<barrier_action>(localities, std::string("testHeterogeneous"));
        // }
        // p["testHeterogeneous"].get_future().get();

    }

    std::string genName(int source, int target)
    {
        std::string basename = "/0_fixme_HPXSimulatorUpdateGroupSdfafafasdasd";

        return basename + "/PatchLink/" +
            StringOps::itoa(source) + "-" +
            StringOps::itoa(target);
    }

    void testFoo()
    {
        typedef RecursiveBisectionPartition<2> PartitionType;
        typedef LibGeoDecomp::HiParSimulator::VanillaStepper<TestCell<2> > StepperType;
        typedef HpxSimulator::UpdateGroup<TestCell<2> > UpdateGroupType;
        typedef StepperType::GridType GridType;
        typedef typename StepperType::PatchProviderVec PatchProviderVec;
        typedef typename StepperType::PatchAccepterVec PatchAccepterVec;

        using namespace boost::assign;

        // good
        // std::string basename = "HPXSimulatorUpdateGroupSdfafafasdasd";

        std::string basename = "/0_HPXSimulatorUpdateGroupSdfafafaXXX";

        // bad
        // std::string basename = "HPXSimulatorUpdateGroupSdfafa";

        // bad
        // std::string basename = "HPXSimulatorUpdateGroup";

        // bad
        // std::string basename = "HPXSimulatorTesfd";
        boost::shared_ptr<UpdateGroupType> updateGroup;
        boost::shared_ptr<PartitionType> partition;
        boost::shared_ptr<Initializer<TestCell<2> > > init;

        rank = hpx::get_locality_id();
        Coord<2> dimensions = Coord<2>(160, 90);
        std::vector<std::size_t> weights;
        weights << 3600
                << 3600
                << 3600
                << 3600;
        std::cout << "weights: " << weights << "\n";
        partition.reset(new PartitionType(Coord<2>(), dimensions, 0, weights));
        int ghostZoneWidth = 1;
        init.reset(new TestInitializer<TestCell<2> >(dimensions));
        updateGroup.reset(
            new UpdateGroupType(
                partition,
                CoordBox<2>(Coord<2>(), dimensions),
                ghostZoneWidth,
                init,
                reinterpret_cast<StepperType*>(0),
                PatchAccepterVec(),
                PatchAccepterVec(),
                PatchProviderVec(),
                PatchProviderVec(),
                basename));

        std::deque<std::size_t> expectedNanoSteps;
        expectedNanoSteps += 5, 7, 8, 33, 55;
        boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter(new MockPatchAccepter<GridType>());
        for (std::deque<std::size_t>::iterator i = expectedNanoSteps.begin();
             i != expectedNanoSteps.end();
             ++i) {
            mockPatchAccepter->pushRequest(*i);
        }
        updateGroup->addPatchAccepter(mockPatchAccepter, StepperType::INNER_SET);

        updateGroup->update(100);
    }

    void testCreation()
    {
        typedef RecursiveBisectionPartition<2> PartitionType;
        typedef LibGeoDecomp::HiParSimulator::VanillaStepper<TestCell<2> > StepperType;
        typedef HpxSimulator::UpdateGroup<TestCell<2> > UpdateGroupType;
        typedef StepperType::GridType GridType;
        typedef typename StepperType::PatchProviderVec PatchProviderVec;
        typedef typename StepperType::PatchAccepterVec PatchAccepterVec;

        using namespace boost::assign;

        std::string basename = "/0_HpxSimulatorTestCreation";
        boost::shared_ptr<UpdateGroupType> updateGroup;
        boost::shared_ptr<PartitionType> partition;
        boost::shared_ptr<Initializer<TestCell<2> > > init;

        rank = hpx::get_locality_id();
        Coord<2> dimensions = Coord<2>(160, 90);
        std::vector<std::size_t> weights;
        weights << 3600
                << 3600
                << 3600
                << 3600;
        std::cout << "weights: " << weights << "\n";
        partition.reset(new PartitionType(Coord<2>(), dimensions, 0, weights));
        int ghostZoneWidth = 1;
        init.reset(new TestInitializer<TestCell<2> >(dimensions));
        updateGroup.reset(
            new UpdateGroupType(
                partition,
                CoordBox<2>(Coord<2>(), dimensions),
                ghostZoneWidth,
                init,
                reinterpret_cast<StepperType*>(0),
                PatchAccepterVec(),
                PatchAccepterVec(),
                PatchProviderVec(),
                PatchProviderVec(),
                basename));

        updateGroup->update(100);

        // std::deque<std::size_t> actualNanoSteps = mockPatchAccepter->getOfferedNanoSteps();
        // expectedNanoSteps.clear();
        // expectedNanoSteps += 5, 7, 8, 33, 55;

        // TS_ASSERT_EQUALS(actualNanoSteps, expectedNanoSteps);
    }



    // xxx
    std::vector<std::size_t> genWeights(
        const unsigned& width,
        const unsigned& height,
        const unsigned& size)
    {
        std::vector<std::size_t> ret(size);
        unsigned totalSize = width * height;
        for (std::size_t i = 0; i < ret.size(); ++i) {
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        }

        return ret;
    }

    // xxx
    long pos(const unsigned& i, const unsigned& size, const unsigned& totalSize)
    {
        return i * totalSize / size;
    }

    void removeFile(std::string name)
    {
        remove(name.c_str());
    }

    void removeFiles(int timestep)
    {
        std::stringstream buf;
        buf << "game." << std::setfill('0') << std::setw(5) << timestep;
        removeFile(buf.str() + ".bov");
        removeFile(buf.str() + ".data");
    }

    std::size_t rank;
    std::vector<hpx::id_type> localities;
    int outputFrequency;
    int maxTimeSteps;
};

}
