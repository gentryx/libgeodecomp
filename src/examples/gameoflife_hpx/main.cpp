
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/io/hpxwritercollector.h>
#include <libgeodecomp/io/serialbovwriter.h>

#include <hpx/hpx_init.hpp>

#include <libgeodecomp/config.h>
#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/stencils.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

class ConwayCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;

    class API : public CellAPITraits::Base
    {};

    static inline unsigned nanoSteps()
    {
        return 1;
    }

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
    CellInitializer() : SimpleInitializer<ConwayCell>(Coord<2>(160, 90), 800)
    {}

    virtual void grid(GridBase<ConwayCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        SuperVector<Coord<2> > startCells;
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


        for (SuperVector<Coord<2> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i) {
            if (rect.inBounds(*i)) {
                ret->at(*i) = ConwayCell(true);
            }
        }
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<SimpleInitializer<ConwayCell> >(*this);
    }
};

class CellToColor {
public:
    Color operator()(const ConwayCell& cell)
    {
        int val = (int)cell.alive * 255;
        return Color(val, val, val);
    }
};

class StateSelector
{
public:
    typedef double VariableType;

    void operator()(const ConwayCell& in, double *out) const
    {
        *out = in.alive;
    }

    static std::string varName()
    {
        return "alive";
    }

    static int dataComponents()
    {
        return 1;
    }

    static std::string dataFormat()
    {
        return "DOUBLE";
    }
};

typedef
    HpxSimulator::HpxSimulator<ConwayCell, HiParSimulator::RecursiveBisectionPartition<2> >
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

typedef LibGeoDecomp::TracingWriter<ConwayCell> TracingWriterType;
BOOST_CLASS_EXPORT_GUID(TracingWriterType, "TracingWriterConwayCell");

typedef LibGeoDecomp::SerialBOVWriter<ConwayCell, StateSelector> BovWriterType;
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

int hpx_main()
{
    {
        int outputFrequency = 1;
        CellInitializer *init = new CellInitializer();

        SimulatorType sim(
            init,
            1, // overcommitFactor
            new TracingBalancer(new OozeBalancer()),
            10, // balancingPeriod
            1 // ghostZoneWidth
            );

        HpxWriterCollectorType::SinkType sink(
            new BovWriterType("game", outputFrequency),
            sim.numUpdateGroups());

        sim.addWriter(
            new HpxWriterCollectorType(
                sink
            ));

        sim.addWriter(
            new TracingWriterType(
                1,
                init->maxSteps()));

        sim.run();
    }
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
