#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/imagepainter.h>
#include <libgeodecomp/io/testcellplotter.h>
#include <libgeodecomp/storage/image.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestCellPlotterTest : public CxxTest::TestSuite
{
public:
    void TestCELLell()
    {
        TestCell<2> cell;
        cell.testValue = 65;
        Image actual(100, 100, Color(100, 100, 100));
        ImagePainter painter(&actual);
        painter.moveTo(Coord<2>(10, 20));
        TestCellPlotter()(cell, painter, Coord<2>(30, 40));

        Image expected(100, 100, Color(100, 100, 100));
        expected.paste(Coord<2>(10, 20), Image(30, 40, Color(65, 47, 11)));

        TS_ASSERT_EQUALS(actual, expected);
    }
};

}
