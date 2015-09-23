#include <cxxtest/TestSuite.h>
#include <iostream>
#include <libflatarray/short_vec.hpp>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/parallelization/openmpsimulator.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasPredefinedMPIDataType<double>
    {};

    inline explicit Cell(double v = 0) :
        temp(v)
    {}

    template<typename NEIGHBORHOOD>
    static void updateLineX(Cell *target, long *x, long endX, const NEIGHBORHOOD& hood, const int nanoStep)
    {
        LIBFLATARRAY_LOOP_PEELER(double, 16, long, x, endX, updateLineImplmentation, target, hood, nanoStep);
    }

    template<typename DOUBLE, typename NEIGHBORHOOD>
    static void updateLineImplmentation(
        long *x,
        long endX,
        Cell *target,
        const NEIGHBORHOOD& hood,
        const int /* nanoStep */)
    {
        // DOUBLE oneSixth = 1.0 / 6.0;

        // for (; *x < (endX - DOUBLE::ARITY + 1); *x += DOUBLE::ARITY) {
        for (; *x < (endX - DOUBLE::ARITY + 1); *x += 1) {
            // DOUBLE buf = &hood[FixedCoord< 0,  0, -1>()].temp;
            // buf += &hood[FixedCoord< 0,  -1,  0>()].temp;
            // buf += &hood[FixedCoord<-1,   0,  0>()].temp;
            // buf += &hood[FixedCoord< 1,   0,  0>()].temp;
            // buf += &hood[FixedCoord< 0,   1,  0>()].temp;
            // buf += &hood[FixedCoord< 0,   0,  1>()].temp;
            // buf *= oneSixth;
            // &target[*x].temp << buf;
            // fixme
            target[*x].temp = exp(asin(sqrt(hood[FixedCoord< 0,   0,  1>()].temp) * target[*x].temp + 1000));
        }
    }

    double temp;
};

class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    explicit CellInitializer(double num) : SimpleInitializer<Cell>(
        Coord<3>(128 * num, 128 * num, 128 * num),
        100)
    {}

    virtual void grid(GridBase<Cell, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        Coord<3> offset =
            Coord<3>::diagonal(gridDimensions().x() * 5 / 128);
        int size = gridDimensions().x() * 50 / 128;


        for (int z = 0; z < size; ++z) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    Coord<3> c = offset + Coord<3>(x, y, z);
                    if (box.inBounds(c)) {
                        ret->set(c, Cell(0.99999999999));
                    }
                }
            }
        }
    }
};

namespace LibGeoDecomp {

class OpenMPSimulatorTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        std::cout << "bäm!\n";

        int outputFrequency = 100;
        CellInitializer *init = new CellInitializer(2.0);
        OpenMPSimulator<Cell> sim(init);
        sim.addWriter(new TracingWriter<Cell>(outputFrequency, init->maxSteps()));

        sim.run();

    }
};

}
