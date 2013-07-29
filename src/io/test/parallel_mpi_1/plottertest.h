#include <ctime>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestCellPlotter
{
public:
    inline void plotCell(
        const TestCell<2>& c,
        Image *img,
        const Coord<2>& upperLeft,
        const unsigned& width,
        const unsigned& height) const
    {
        int value = 1 + c.pos.x() + c.pos.y() * c.dimensions.dimensions.x();
        img->fillBox(upperLeft, width, height, Color(47, 11, value));
    }
};


class PlotterTest : public CxxTest::TestSuite
{
private:
    unsigned width;
    unsigned height;
    Plotter<TestCell<2>, TestCellPlotter> *plotter;
    TestCellPlotter simplePlotter;

public:
    void setUp()
    {
        width = 10;
        height = 24;
        plotter = new Plotter<TestCell<2>, TestCellPlotter>(&simplePlotter, width, height);
    }


    void tearDown()
    {
        delete plotter;
    }


    void testPlotGridDimensions()
    {
        unsigned gridDimX = 2;
        unsigned gridDimY = 3;
        int expectedDimX = gridDimX * width;
        int expectedDimY = gridDimY * height;

        Grid<TestCell<2> > testGrid(Coord<2>(gridDimX, gridDimY));
        Image result = plotter->plotGrid(testGrid);

        TS_ASSERT_EQUALS(result.getDimensions().x(), expectedDimX);
        TS_ASSERT_EQUALS(result.getDimensions().y(), expectedDimY);
    }


    void testPlotGridContent()
    {
        unsigned gridDimX = 2;
        unsigned gridDimY = 3;
        unsigned expectedDimX = gridDimX * width;
        unsigned expectedDimY = gridDimY * height;

        Grid<TestCell<2> > testGrid(Coord<2>(gridDimX, gridDimY));
        testGrid[Coord<2>(0, 0)].testValue = 33;
        testGrid[Coord<2>(1, 0)].testValue =  66;
        testGrid[Coord<2>(0, 1)].testValue = 100;
        testGrid[Coord<2>(1, 1)].testValue = 133;
        testGrid[Coord<2>(0, 2)].testValue = 166;
        testGrid[Coord<2>(1, 2)].testValue = 200;

        Image result = plotter->plotGrid(testGrid);

        Image expected(expectedDimX, expectedDimY);

        Image f(width, height);
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                Coord<2> pos(x, y);
                simplePlotter.plotCell(testGrid[pos], &f, Coord<2>(), width, height);
                expected.paste(Coord<2>(x * width, y * height), f);
            }
        }

        TS_ASSERT_EQUALS(expected, result);
    }



    void testPlotGridInViewportUpperLeft()
    {
        Grid<TestCell<2> > testGrid(Coord<2>(2, 3));
        testGrid[Coord<2>(0, 0)].testValue =  33;
        testGrid[Coord<2>(1, 0)].testValue =  66;
        testGrid[Coord<2>(0, 1)].testValue = 100;
        testGrid[Coord<2>(1, 1)].testValue = 133;
        testGrid[Coord<2>(0, 2)].testValue = 166;
        testGrid[Coord<2>(1, 2)].testValue = 200;
        Image uncut = plotter->plotGrid(testGrid);

        int x = 15;
        int y = 10;
        int width = 40;
        int height = 80;

        int colorWidth = uncut.getDimensions().x() - x;
        int colorHeight = uncut.getDimensions().y() - y;
        int blackWidth = width - colorWidth;
        int blackHeight = height - colorHeight;

        Image actual = plotter->plotGridInViewport(testGrid, Coord<2>(x, y),
                                              width, height);
        /* check dimensions */
        TS_ASSERT_EQUALS(actual.getDimensions().x(), width);
        TS_ASSERT_EQUALS(actual.getDimensions().y(), height);

        /* check right portion */
        TS_ASSERT_EQUALS(actual.slice(Coord<2>(colorWidth, 0), blackWidth, height),
                         Image(blackWidth, height, Color::BLACK));

        /* check lower portion */
        TS_ASSERT_EQUALS(actual.slice(Coord<2>(0, colorHeight), width, blackHeight),
                         Image(width, blackHeight, Color::BLACK));

        /* check upper left corner */
        Image actSlice = actual.slice(Coord<2>(0, 0), colorWidth, colorHeight);
        Image uncSlice = uncut.slice(Coord<2>(x, y), colorWidth, colorHeight);
        TS_ASSERT_EQUALS(actSlice, uncSlice);
    }


    void testPlotGridInViewportLarge()
    {
        Grid<TestCell<2> > testGrid(Coord<2>(2000, 2000));
        int t1 = time(0);
        Image actual = plotter->plotGridInViewport(testGrid, Coord<2>(500, 500),
                                              1000, 1000);
        int t2 = time(0);
        int span = t2 - t1;
        TS_ASSERT(span < 10);
    }


    void testSetGetDimensions()
    {
        Coord<2> dims(42, 11);
        plotter->setCellDimensions(dims[0], dims[1]);
        TS_ASSERT_EQUALS(dims, plotter->getCellDimensions());
    }

};

}
