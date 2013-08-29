#include <ctime>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/imagepainter.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/testcellplotter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PlotterTest : public CxxTest::TestSuite
{
private:
    unsigned width;
    unsigned height;
    Plotter<TestCell<2>, TestCellPlotter> *plotter;
public:
    void setUp()
    {
        width = 10;
        height = 24;
        plotter = new Plotter<TestCell<2>, TestCellPlotter>(Coord<2>(width, height));
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
        Coord<2> gridDim(gridDimX, gridDimY);

        Grid<TestCell<2> > testGrid(gridDim);
        Image result(plotter->calcImageDim(gridDim));
        plotter->plotGrid(testGrid, ImagePainter(&result));

        TS_ASSERT_EQUALS(result.getDimensions().x(), expectedDimX);
        TS_ASSERT_EQUALS(result.getDimensions().y(), expectedDimY);
    }


    void testPlotGridContent()
    {
        unsigned gridDimX = 2;
        unsigned gridDimY = 3;
        unsigned expectedDimX = gridDimX * width;
        unsigned expectedDimY = gridDimY * height;
        Coord<2> gridDim(gridDimX, gridDimY);

        Grid<TestCell<2> > testGrid(Coord<2>(gridDimX, gridDimY));
        testGrid[Coord<2>(0, 0)].testValue = 33;
        testGrid[Coord<2>(1, 0)].testValue =  66;
        testGrid[Coord<2>(0, 1)].testValue = 100;
        testGrid[Coord<2>(1, 1)].testValue = 133;
        testGrid[Coord<2>(0, 2)].testValue = 166;
        testGrid[Coord<2>(1, 2)].testValue = 200;

        Image result(plotter->calcImageDim(gridDim));
        plotter->plotGrid(testGrid, ImagePainter(&result));

        Image expected(expectedDimX, expectedDimY);

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                Coord<2> pos(x, y);
                ImagePainter painter(&expected);
                painter.moveTo(Coord<2>(x * width, y * height));

                TestCellPlotter()(testGrid[pos], painter, Coord<2>(width, height));
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
        Image uncut(plotter->calcImageDim(Coord<2>(2, 3)));
        plotter->plotGrid(testGrid, ImagePainter(&uncut));

        int x = 15;
        int y = 10;
        int width = 40;
        int height = 80;

        int colorWidth = uncut.getDimensions().x() - x;
        int colorHeight = uncut.getDimensions().y() - y;
        int blackWidth = width - colorWidth;
        int blackHeight = height - colorHeight;

        Image actual(Coord<2>(width, height));

        plotter->plotGridInViewport(
            testGrid,
            ImagePainter(&actual),
            CoordBox<2>(Coord<2>(x, y), Coord<2>(width, height)));
        TS_ASSERT_EQUALS(actual.getDimensions().x(), width);
        TS_ASSERT_EQUALS(actual.getDimensions().y(), height);

        // check right portion
        TS_ASSERT_EQUALS(actual.slice(Coord<2>(colorWidth, 0), blackWidth, height),
                         Image(blackWidth, height, Color::BLACK));

        // check lower portion
        TS_ASSERT_EQUALS(actual.slice(Coord<2>(0, colorHeight), width, blackHeight),
                         Image(width, blackHeight, Color::BLACK));

        // check upper left corner
        Image actSlice = actual.slice(Coord<2>(0, 0), colorWidth, colorHeight);
        Image uncSlice = uncut.slice(Coord<2>(x, y), colorWidth, colorHeight);
        TS_ASSERT_EQUALS(actSlice, uncSlice);
    }

    void testPlotGridInViewportLarge()
    {
        Grid<TestCell<2> > testGrid(Coord<2>(2000, 2000));
        int t1 = time(0);
        Image actual(Coord<2>(1000, 1000));
        plotter->plotGridInViewport(
            testGrid,
            ImagePainter(&actual),
            CoordBox<2>(Coord<2>(500, 500), Coord<2>(1000, 1000)));
        int t2 = time(0);
        int span = t2 - t1;
        TS_ASSERT(span < 10);
    }
};

}
