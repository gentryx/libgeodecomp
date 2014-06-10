#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/io/qtwidgetwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>

#ifdef LIBGEODECOMP_WITH_QT

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

#include <QtGui/QApplication>

#ifdef __ICC
#pragma warning pop
#endif

#endif

using namespace LibGeoDecomp;

class MyQtTestCell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<2, 1> >
    {};

    inline explicit MyQtTestCell(double v = 0) :
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

class MyQtCellColorConverter
{
public:
    Color operator()(const MyQtTestCell& cell)
    {
        return Color(255, cell.temp * 255, 0);
    }
};

namespace LibGeoDecomp {

class QtWidgetWriterTest : public CxxTest::TestSuite
{
public:
    void testWithPalette()
    {
#ifdef LIBGEODECOMP_WITH_QT
        int argc = 0;
        char **argv = 0;
        QApplication app(argc, argv);

        Coord<2> cellDim(10, 10);
        Palette<double> palette;
        palette.addColor(0.0, Color::RED);
        palette.addColor(1.0, Color::YELLOW);
        TS_ASSERT_EQUALS(palette[0.0], Color::RED);
        TS_ASSERT_EQUALS(palette[1.0], Color::YELLOW);

        QtWidgetWriter<MyQtTestCell> writer(&MyQtTestCell::temp, palette, cellDim);
        QWidget *widget = writer.widget();
        widget->resize(1200, 900);

        Grid<MyQtTestCell> grid(Coord<2>(100, 50));
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 100; ++x) {
                double xReal = x / 100.0;
                double yReal = y / 50.0;
                double value = (xReal * xReal + yReal * yReal) * 0.5;
                grid[Coord<2>(x, y)] = MyQtTestCell(value);
            }
        }
        writer.stepFinished(grid, 12, WRITER_INITIALIZED);

        QImage& image = writer.widget()->curImage;
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 100; ++x) {
                for (int tileY = 0; tileY < cellDim.y(); ++tileY) {
                    for (int tileX = 0; tileX < cellDim.x(); ++tileX) {
                        int logicalX = x * cellDim.x() + tileX;
                        int logicalY = y * cellDim.y() + tileY;

                        double xReal = x / 100.0;
                        double yReal = y / 50.0;
                        int value = (xReal * xReal + yReal * yReal) * 0.5 * 255;

                        // blue
                        TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 0], 0);
                        // green
                        TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 1], value);
                        // red (account for rounding errors)
                        TS_ASSERT(image.scanLine(logicalY)[4 * logicalX + 2] >= 254);
                        TS_ASSERT(image.scanLine(logicalY)[4 * logicalX + 2] <= 255);
                        // alpha
                        TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 3], 255);
                    }
                }
            }
        }
#endif
    }

    void testWithQuickPalette()
    {
#ifdef LIBGEODECOMP_WITH_QT
        int argc = 0;
        char **argv = 0;
        QApplication app(argc, argv);

        Coord<2> cellDim(10, 20);

        QtWidgetWriter<MyQtTestCell> writer(&MyQtTestCell::temp, -10.0, 10.0, cellDim);
        QWidget *widget = writer.widget();
        widget->resize(1000, 1000);

        Grid<MyQtTestCell> grid(Coord<2>(10, 20));
        for (int y = 0; y < 20; ++y) {
            for (int x = 0; x < 10; ++x) {
                double xReal = x / 9.0;
                double yReal = y / 19.0;
                double value = (xReal + yReal) * 10 - 10;
                grid[Coord<2>(x, y)] = MyQtTestCell(value);
            }
        }
        writer.stepFinished(grid, 1234, WRITER_INITIALIZED);

        QImage& image = writer.widget()->curImage;

#define CHECK_TILE(X, Y, RED, GREEN, BLUE)                              \
        for (int tileY = 0; tileY < cellDim.y(); ++tileY) {             \
            for (int tileX = 0; tileX < cellDim.x(); ++tileX) {         \
                int logicalX = X * cellDim.x() + tileX;                 \
                int logicalY = Y * cellDim.y() + tileY;                 \
                                                                        \
                TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 0], BLUE); \
                TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 1], GREEN); \
                TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 2], RED); \
                TS_ASSERT_EQUALS(image.scanLine(logicalY)[4 * logicalX + 3], 255); \
            }                                                           \
        }

        // upper left tile should be all blue:
        CHECK_TILE(0,  0,   0,   0, 255);
        // lower right tile should be red:
        CHECK_TILE(9, 19, 255,   0,   0);
        // opposite corners should be green:
        CHECK_TILE(0, 19,   0, 255,   0);
        CHECK_TILE(9,  0,   0, 255,   0);

#undef CHECK_TILE

#endif
    }
};

}
