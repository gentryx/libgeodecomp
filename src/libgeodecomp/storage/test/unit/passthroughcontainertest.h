#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/boxcell.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/storage/passthroughcontainer.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class Heater
{
public:
    explicit
    Heater(double energy = 0.0) :
        energy(energy)
    {}

    FloatCoord<2> getPos() const
    {
        return FloatCoord<2>();
    }

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        // intentionally left blank
    }

    double energy;
};

class GridCell
{
public:
    friend class PassThroughContainerTest;

    explicit
    GridCell(double temperature = 21.0) :
        temperature(temperature)
    {}

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        temperature = (hood.cells[Coord<2>( 0, -1)].temperature +
                       hood.cells[Coord<2>(-1,  0)].temperature +
                       hood.cells[Coord<2>( 1,  0)].temperature +
                       hood.cells[Coord<2>( 0,  1)].temperature) * 0.25;
        addHeaters(hood.heaters.begin(), hood.heaters.end());
    }

private:
    double temperature;

    template<typename ITER>
    void addHeaters(const ITER& begin, const ITER& end)
    {
        for (ITER i = begin; i != end; ++i) {
            temperature += i->energy;
        }
    }
};

DECLARE_MULTI_CONTAINER_CELL(
    TestContainer,
    TestContainer,
    (((BoxCell<FixedArray<Heater, 10> >))(heaters))
    (((PassThroughContainer<GridCell>))(cells)) )

class PassThroughContainerTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        Coord<2> gridDim(10, 5);
        Region<2> region;
        region << CoordBox<2>(Coord<2>(0, 0), gridDim);

        Grid<TestContainer> gridOld(gridDim);
        Grid<TestContainer> gridNew(gridDim);
        gridOld[Coord<2>(0, 0)].heaters << Heater(1.0);
        gridOld[Coord<2>(0, 0)].heaters << Heater(2.0);
        gridOld[Coord<2>(1, 0)].heaters << Heater(3.0);
        gridOld[Coord<2>(5, 3)].heaters << Heater(5.0);

        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 10; ++x) {
                Coord<2> c(x, y);
                gridOld[c].cells = GridCell(10.0);
            }
        }

        UpdateFunctor<TestContainer>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);

        double expected;

        // left and upper neighbor of (0, 0) have the default value of
        // 21.0 (because they're out of grid), The lower and right
        // neighbor are at 10.0 (because we initialized them so). The
        // first 3 heaters should be visible (values 1.0, 2.0, 3.0)
        expected = (21.0 * 2 + 10 * 2) * 0.25 + 1.0 + 2.0 + 3.0;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(0, 0)].cells.cell.temperature);

        // For coordinate (1, 0) we have only 1 out of bounds neighbor
        // (up) and again 3 heaters:
        expected = (21.0 * 1 + 10 * 3) * 0.25 + 1.0 + 2.0 + 3.0;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(1, 0)].cells.cell.temperature);

        // Coord (2, 0) won't see the first two heaters which are
        // located at (0, 0):
        expected = (21.0 * 1 + 10 * 3) * 0.25 + 3.0;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(2, 0)].cells.cell.temperature);

        // Coord (3, 0) won't see any heater:
        expected = (21.0 * 1 + 10 * 3) * 0.25;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(3, 0)].cells.cell.temperature);

        // Coord (3, 1) won't see any heater, nor any border cell:
        expected = (10.0 * 4) * 0.25;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(3, 1)].cells.cell.temperature);

        // Coords (4, 3) through (6, 3) see the heater with value 5.0:
        expected = (10.0 * 4) * 0.25 + 5.0;
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(4, 3)].cells.cell.temperature);
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(5, 3)].cells.cell.temperature);
        TS_ASSERT_EQUALS(expected, gridNew[Coord<2>(6, 3)].cells.cell.temperature);
    }
};

}
