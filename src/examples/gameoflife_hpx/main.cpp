#include <libgeodecomp.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <hpx/hpx_init.hpp>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

using namespace LibGeoDecomp;

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

    void update(const CoordMap<ConwayCell>& neighborhood, unsigned)
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

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(ConwayCell)

class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    CellInitializer() : SimpleInitializer<ConwayCell>(Coord<2>(160, 90), 800)
    {}

    virtual void grid(GridBase<ConwayCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        std::vector<Coord<2> > startCells;
        // start with a single glider...
        //          x
        //           x
        //         xxx
        startCells << Coord<2>(11, 10)
                   << Coord<2>(12, 11)
                   << Coord<2>(10, 12) << Coord<2>(11, 12) << Coord<2>(12, 12);


        // ...add a Diehard pattern...
        //                x
        //          xx
        //           x   xxx
        startCells << Coord<2>(55, 70) << Coord<2>(56, 70)
                   << Coord<2>(56, 71)
                   << Coord<2>(60, 71) << Coord<2>(61, 71) << Coord<2>(62, 71)
                   << Coord<2>(61, 69);

        // ...and an Acorn pattern:
        //        x
        //          x
        //       xx  xxx
        startCells << Coord<2>(111, 30)
                   << Coord<2>(113, 31)
                   << Coord<2>(110, 32) << Coord<2>(111, 32)
                   << Coord<2>(113, 32) << Coord<2>(114, 32) << Coord<2>(115, 32);


        for (std::vector<Coord<2> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i) {
            if (rect.inBounds(*i)) {
                ret->set(*i, ConwayCell(true));
            }
        }
    }
};

typedef HpxSimulator<ConwayCell, RecursiveBisectionPartition<2> > SimulatorType;

typedef TracingWriter<ConwayCell> TracingWriterType;

int hpx_main()
{
    {
        CellInitializer *init = new CellInitializer();

        SimulatorType sim(
            init,
            // number an relative speed of update groups:
            std::vector<double>(1, 1.0),
            new TracingBalancer(new OozeBalancer()),
            // load balancing period:
            100,
            // ghost zone width:
            1);

        sim.addWriter(
            new TracingWriterType(
                1,
                init->maxSteps(),
                1));

        sim.run();
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
