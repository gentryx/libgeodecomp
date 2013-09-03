#include <libgeodecomp/config.h>
#include <libgeodecomp/io/qtwidgetwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>

#ifdef LIBGEODECOMP_FEATURE_QT
#include <QtGui/QApplication>
#endif

using namespace LibGeoDecomp;

class MyQtTestCell
{
public:
    class API :
        public CellAPITraits::Base,
        public CellAPITraitsFixme::HasStencil<Stencils::VonNeumann<2, 1> >,
        public CellAPITraitsFixme::HasCubeTopology<2>
    {};

    inline MyQtTestCell(double v = 0) :
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
    void testBasic()
    {
#ifdef LIBGEODECOMP_FEATURE_QT
        int argc = 0;
        char **argv = 0;
        QApplication app(argc, argv);

        Coord<2> cellDim(10, 10);
        QtWidgetWriter<MyQtTestCell, SimpleCellPlotter<MyQtTestCell, MyQtCellColorConverter> > writer(cellDim);
        QWidget *widget = writer.widget();
        widget->resize(1200, 900);

        Grid<MyQtTestCell> grid(Coord<2>(100, 50));
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 100; ++x) {
                double xReal = x * (1.0 / 100.0);
                double yReal = y * (1.0 / 50.0);
                double value = (xReal * xReal + yReal * yReal) * 0.5;
                grid[Coord<2>(x, y)] = value;
            }
        }
        writer.stepFinished(grid, 12, WRITER_INITIALIZED);

        QImage *image = writer.widget()->getImage();
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 100; ++x) {
                for (int tileY = 0; tileY < cellDim.y(); ++tileY) {
                    for (int tileX = 0; tileX < cellDim.y(); ++tileX) {
                        int logicalX = x * cellDim.x() + tileX;
                        int logicalY = y * cellDim.y() + tileY;

                        double xReal = x * (1.0 / 100.0);
                        double yReal = y * (1.0 / 50.0);
                        int value = (xReal * xReal + yReal * yReal) * 0.5 * 255;

                        // blue
                        TS_ASSERT_EQUALS(image->scanLine(logicalY)[4 * logicalX + 0], 0);
                        // green
                        TS_ASSERT_EQUALS(image->scanLine(logicalY)[4 * logicalX + 1], value);
                        // red
                        TS_ASSERT_EQUALS(image->scanLine(logicalY)[4 * logicalX + 2], 255);
                        // alpha
                        TS_ASSERT_EQUALS(image->scanLine(logicalY)[4 * logicalX + 3], 255);
                    }
                }
            }
        }
#endif
    }
};

}
