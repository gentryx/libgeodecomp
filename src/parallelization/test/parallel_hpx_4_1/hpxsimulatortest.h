#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

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
LIBGEODECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(
    SimulatorType,
    ConwayCellSimulator
)
LIBGEODECOMP_REGISTER_HPX_SIMULATOR(
    SimulatorType,
    ConwayCellSimulator
)

BOOST_CLASS_EXPORT_GUID(CellInitializer, "CellInitializer");

typedef LibGeoDecomp::SerialBOVWriter<ConwayCell> BovWriterType;
BOOST_CLASS_EXPORT_GUID(BovWriterType, "BovWriterConwayCell");

typedef
    LibGeoDecomp::HpxWriterCollector<ConwayCell>
    HpxWriterCollectorType;

LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR_DECLARATION(
    HpxWriterCollectorType,
    ConwayCellWriterCollector
)
LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR(
    HpxWriterCollectorType,
    ConwayCellWriterCollector
)

namespace LibGeoDecomp {

class HpxSimulatorTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        outputFrequency = 1;
        maxTimeSteps = 10;
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
        CellInitializer *init = new CellInitializer(maxTimeSteps);

        SimulatorType sim(
            init,
            1, // overcommitFactor
            new TracingBalancer(new OozeBalancer()),
            10, // balancingPeriod
            1 // ghostZoneWidth
                          );

        HpxWriterCollectorType::SinkType sink(
            new BovWriterType(&ConwayCell::alive, "game", outputFrequency),
            sim.numUpdateGroups());

        sim.addWriter(new HpxWriterCollectorType(sink));

        sim.run();
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

    int outputFrequency;
    int maxTimeSteps;

};

}
