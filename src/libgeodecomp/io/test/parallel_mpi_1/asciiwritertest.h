#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/asciiwriter.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/testinitializer.h>

#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cxxtest/TestSuite.h>
#include <stdio.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ASCIIWriterTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        tempFile = TempFile::parallel("libGeoDecompTempfile");
        simulator = new SerialSimulator<TestCell<2> >(new TestInitializer<TestCell<2> >(Coord<2>(2, 3)));
    }

    void tearDown()
    {
        delete simulator;

        for (int i = 0; i < 100; i++) {
            std::ostringstream f;
            f << tempFile << "." << std::setw(4) << std::setfill('0') << i << ".ascii";
            remove(f.str().c_str());
        }
    }

    void testWriteASCII()
    {
        simulator->addWriter(
            new ASCIIWriter<TestCell<2> >(tempFile, &TestCell<2>::testValue));
        simulator->run();

        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << tempFile << "." << std::setfill('0')
                     << std::setw(4) << i << ".ascii";
            TS_ASSERT_FILE(filename.str());
        }

        std::string firstFile = tempFile + ".0002.ascii";
        std::ifstream infile(firstFile.c_str());

        std::string expected = "\n\n1 2 \n3 4 \n5 6 ";
        std::ostringstream content;

        for (unsigned i = 0; i < expected.length(); i++) {
            content << (char)infile.get();
        }

        TS_ASSERT_EQUALS(content.str(), expected);
    }

    void testWriteASCIIEveryN()
    {
        int period = 2;
        simulator->addWriter(new ASCIIWriter<TestCell<2> >(tempFile, &TestCell<2>::testValue, period));
        simulator->run();

        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << tempFile << "." << std::setfill('0') << std::setw(4)
                     << i << ".ascii";
            if (i % period == 0) {
                TS_ASSERT_FILE(filename.str());
            } else {
                TS_ASSERT_NO_FILE(filename.str());
            }
        }
    }

    void testFileOpenError()
    {
        std::string path("/non/existent/path/prefix2");
        std::string expectedErrorMessage("Could not open file " + path);

        ASCIIWriter<TestCell<2> > *writer = new ASCIIWriter<TestCell<2> >(path, &TestCell<2>::testValue);
        simulator->addWriter(writer);

        TS_ASSERT_THROWS_ASSERT(
            writer->stepFinished(*simulator->getGrid(), simulator->getStep(), WRITER_INITIALIZED),
            FileOpenException& exception,
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
