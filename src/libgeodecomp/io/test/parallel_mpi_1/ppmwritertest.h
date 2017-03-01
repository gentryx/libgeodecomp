#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/testinitializer.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestCellPalette
{
public:
    Color operator[](const double value) const
    {
        return Color(value, 47.0, 11.0);
    }
};

class PPMWriterTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        tempFile = TempFile::serial("libGeoDecompTempfile");
        simulator = new SerialSimulator<TestCell<2> >(
            new TestInitializer<TestCell<2> >(Coord<2>(10, 11)));
    }


    void tearDown() {
        delete simulator;
        for (int i = 0; i < 100; i++) {
            std::ostringstream f;
            f << tempFile << "."
              << std::setw(4) << std::setfill('0') << i << ".ppm";
            remove(f.str().c_str());
        }
    }


    void testWritePPM()
    {
        simulator->addWriter(
            new PPMWriter<TestCell<2> >(
                &TestCell<2>::testValue, TestCellPalette(), tempFile, 1, Coord<2>(20, 20)));
        simulator->run();

        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << tempFile << "." << std::setfill('0') << std::setw(4)
                     << i << ".ppm";
            TS_ASSERT_FILE(filename.str());
        }

        std::string firstFile = tempFile + ".0000.ppm";
        std::ifstream infile(firstFile.c_str());

        std::vector<char> expected;
        expected << 0x50 << 0x36 << 0x20 // P6_
                 << 0x32 << 0x30 << 0x30 << 0x20 // width,
                 << 0x32 << 0x32 << 0x30 << 0x20 // height,
                 << 0x32 << 0x35 << 0x35 << 0x0a; // and maxcolor (space separated)

        // each cell gets plotted as a tile. we check the first lines
        // of the two leftmost cells in the upper grid line
        for (int i = 0; i < 20; i++) {
            expected << 0x01 << 0x2f << 0x0b; // rgb rgb...
        }

        for (int i = 0; i < 20; i++) {
            expected << 0x02 << 0x2f << 0x0b; // rgb rgb...
        }

        std::vector<char> content;
        for (unsigned i = 0; i < expected.size(); i++) {
            content << infile.get();
        }

        TS_ASSERT_EQUALS(content, expected);
    }

    void testWritePPMPeriod()
    {
        int period = 2;
        simulator->addWriter(
            new PPMWriter<TestCell<2> >(
                &TestCell<2>::testValue, TestCellPalette(), tempFile, period));
        simulator->run();

        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << tempFile << "." << std::setfill('0') << std::setw(4)
                     << i << ".ppm";

            if (i % period == 0) {
                TS_ASSERT_FILE(filename.str());
            } else {
                TS_ASSERT_NO_FILE(filename.str());
            }
        }
    }

    void testFileOpenError()
    {
        std::string path("/non/existent/path/prefix1");
        std::string expectedErrorMessage("Could not open file " + path);
        PPMWriter<TestCell<2> > *writer =
            new PPMWriter<TestCell<2> >(
                &TestCell<2>::testValue, TestCellPalette(), path);
        TS_ASSERT_THROWS_ASSERT(
            writer->stepFinished(
                *simulator->getGrid(),
                simulator->getStep(),
                WRITER_INITIALIZED),
            FileOpenException &exception,
            TS_ASSERT_SAME_DATA(
                expectedErrorMessage.c_str(),
                exception.what(),
                expectedErrorMessage.length()));
    }

private:
    std::string tempFile;
    MonolithicSimulator<TestCell<2> > *simulator;
};

}
