#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PPMWriterTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        std::cout << "  ping1\n";
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

        QuickPalette<double> palette = dynamic_cast<
            SimpleCellPlotterHelpers::CellToColor<TestCell<2>, double, QuickPalette<double> >*>(
                &*writer2.plotter.cellPlotter.cellToColorSelector.filter)->palette;

        TS_ASSERT_EQUALS(Color::BLACK,     palette[-1]);
        TS_ASSERT_EQUALS(Color::WHITE,     palette[ 2]);
        TS_ASSERT_EQUALS(Color(0, 0, 255), palette[ 0]);
        TS_ASSERT_EQUALS(Color(255, 0, 0), palette[ 1]);
    }

    void testSerializationOfWriterViaSharedPtr()
    {
        boost::shared_ptr<Writer<TestCell<2> > > writer1(new PPMWriter<TestCell<2> >(
            &TestCell<2>::testValue,
            5.0,
            6.0,
            "bingo",
            47110,
            Coord<2>(1, 2)));

        boost::shared_ptr<Writer<TestCell<2> > > writer2;

        std::vector<char> buffer;
        {
            hpx::serialization::output_archive outputArchive(buffer);

            outputArchive << writer1;
        }
        {
            hpx::serialization::input_archive inputArchive(buffer);
            inputArchive >> writer2;
        }

        PPMWriter<TestCell<2> > *writer3 = dynamic_cast<PPMWriter<TestCell<2> >*>(&*writer2);

        QuickPalette<double> palette = dynamic_cast<
            SimpleCellPlotterHelpers::CellToColor<TestCell<2>, double, QuickPalette<double> >*>(
                &*writer3->plotter.cellPlotter.cellToColorSelector.filter)->palette;

        TS_ASSERT_EQUALS(Color::BLACK,     palette[4]);
        TS_ASSERT_EQUALS(Color::WHITE,     palette[7]);
        TS_ASSERT_EQUALS(Color(0, 0, 255), palette[5]);
        TS_ASSERT_EQUALS(Color(255, 0, 0), palette[6]);
    }
};

}
