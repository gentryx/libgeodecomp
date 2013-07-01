/**
 * Minimal 2D Jacobi example. Code which is commented out demos how to
 * add a PPMWriter for output.
 */
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    typedef Stencils::VonNeumann<2, 1> Stencil;

    typedef Topologies::Cube<2>::Topology Topology;

    class API : public CellAPITraits::Base
    {};


    static inline unsigned nanoSteps()
    {
        return 1;
    }

    inline Cell(const double& v = 0) :
        temp(v)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[Coord<2>( 0, -1)].temp +
                neighborhood[Coord<2>(-1,  0)].temp +
                neighborhood[Coord<2>( 0,  0)].temp +
                neighborhood[Coord<2>( 1,  0)].temp +
                neighborhood[Coord<2>( 0,  1)].temp) * (1.0 / 5.0);
    }

    double temp;
};

class CellInitializer : public SimpleInitializer<Cell>
{
public:
    CellInitializer() : SimpleInitializer<Cell>(Coord<2>(512, 512), 100)
    {}

    virtual void grid(GridBase<Cell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        int offsetX = 10;
        int offsetY = 10;

        for (int y = 0; y < 250; ++y) {
            for (int x = 0; x < 250; ++x) {
                Coord<2> c(x + offsetX, y + offsetY);
                if (rect.inBounds(c)) {
                    ret->at(c) = Cell(0.99999999999);
                }
            }
        }
    }
};

class CellToColor {
public:
    Color operator()(const Cell& cell)
    {
        if (cell.temp < 0) {
            return Color(0, 0, 0);
        }
        if (cell.temp < 0.25) {
            return Color(0, (cell.temp - 0.0) * 1020, 255);
        }
        if (cell.temp < 0.50) {
            return Color(0, 255, 255 - (cell.temp - 0.25) * 1020);
        }
        if (cell.temp < 0.75) {
            return Color((cell.temp - 0.5) * 1020, 255, 0);
        }
        if (cell.temp < 1.00) {
            return Color(255, 255 - (cell.temp - 0.75) * 1020, 0);
        }
        return Color(255, 255, 255);
    }
};

void runSimulation()
{
    SerialSimulator<Cell>
        sim(new CellInitializer());

    int outputFrequency = 1;
    // sim.addWriter(
    //     new PPMWriter<Cell, SimpleCellPlotter<Cell, CellToColor> >(
    //         "./jacobi",
    //         outputFrequency,
    //         1,
    //         1));
    sim.addWriter(new TracingWriter<Cell>(outputFrequency, 100));

    sim.run();
}

int main(int argc, char **argv)
{
    runSimulation();
    return 0;
}
