/**
 * We need to include typemaps first to avoid problems with Intel
 * MPI's C++ bindings (which may collide with stdio.h's SEEK_SET,
 * SEEK_CUR etc.).
 */
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/image.h>

using namespace LibGeoDecomp;

class ConwayCell
{
public:
    class API :
        public APITraits::HasPredefinedMPIDataType<char>
    {};

    explicit ConwayCell(bool alive = false) :
        alive(alive)
    {}

    template<typename COORD_MAP>
    int countLivingNeighbors(const COORD_MAP& neighborhood)
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

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, unsigned)
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
};

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
        startCells << Coord<2>(55, 70) << Coord<2>(56, 70) << Coord<2>(56, 71)
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

class CellToColor {
public:
    Color operator()(const ConwayCell& cell)
    {
        int val = (int)cell.alive * 255;
        return Color(val, val, val);
    }
};

void runSimulation()
{
    int outputFrequency = 1;
    CellInitializer *init = new CellInitializer();

    StripingSimulator<ConwayCell> sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new OozeBalancer()),
        10);

    sim.addWriter(
        new BOVWriter<ConwayCell>(
            Selector<ConwayCell>(&ConwayCell::alive, "alive"),
            "game",
            outputFrequency));

    sim.addWriter(
        new TracingWriter<ConwayCell>(
            1,
            init->maxSteps()));

    sim.run();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    runSimulation();

    MPI_Finalize();
    return 0;
}
