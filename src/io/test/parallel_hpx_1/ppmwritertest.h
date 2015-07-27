#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PPMWriterTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        PPMWriter<TestCell<2> > writer1(
            &TestCell<2>::testValue,
            0.0,
            1.0,
            "bingo",
            4711,
            Coord<2>(1, 2));

        PPMWriter<TestCell<2> > writer2(
            &TestCell<2>::cycleCounter,
            unsigned(20),
            unsigned(30),
            "bongo",
            4712,
            Coord<2>(5, 4));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        QuickPalette<double> palette = static_cast<
            SimpleCellPlotterHelpers::CellToColor<TestCell<2>, double, QuickPalette<double> >*>(
                &*writer2.plotter.cellPlotter.cellToColor.filter)->palette;

        TS_ASSERT_EQUALS(Color::BLACK,     palette[-1]);
        TS_ASSERT_EQUALS(Color::WHITE,     palette[ 2]);
        TS_ASSERT_EQUALS(Color(0, 0, 255), palette[ 0]);
        TS_ASSERT_EQUALS(Color(255, 0, 0), palette[ 1]);
    }

};

}
