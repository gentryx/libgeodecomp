#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cxxtest/TestSuite.h>
#include <stdio.h>
#include <boost/assign/std/vector.hpp>
#include "../../../misc/tempfile.h"
#include "../../../misc/testcell.h"
#include "../../../misc/testhelper.h"
#include "../../../parallelization/serialsimulator.h"
#include "../../asciiwriter.h"
#include "../../ioexception.h"
#include "../../testinitializer.h"

using namespace boost::assign; 
using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TestValueSelector
{
public:
    inline double operator()(const TestCell<2>& c)
    {
        return c.testValue;
    }
};

class ASCIIWriterTest : public CxxTest::TestSuite 
{
public:

    void setUp()
    {
        _tempFile = TempFile::parallel("libGeoDecompTempfile");
        _simulator = new SerialSimulator<TestCell<2> >(
            new TestInitializer<2>(Coord<2>(2, 3)));
        
    }


    void tearDown() {
        delete _simulator;
        for (int i = 0; i < 100; i++) {
            std::ostringstream f;
            f << _tempFile << "." << std::setw(4) << std::setfill('0') << i << ".ascii";
            remove(f.str().c_str());
        }
    }


    void testWriteASCII()
    {
        new ASCIIWriter<TestCell<2>, TestValueSelector>(_tempFile, _simulator);
        _simulator->run();
        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << _tempFile << "." << std::setfill('0') << std::setw(4)
                     << i << ".ascii";
            TS_ASSERT_FILE(filename.str());
        }

        std::string firstFile = _tempFile + ".0002.ascii";
        std::ifstream infile(firstFile.c_str());

        std::string expected = "56\n34\n12\n";
        std::ostringstream content;

        for (unsigned i = 0; i < expected.length(); i++)
            content << (char)infile.get();

        TS_ASSERT_EQUALS(content.str(), expected);
    }


    void testWriteASCIIEveryN()
    {
        int everyN = 2;
        new ASCIIWriter<TestCell<2>, TestValueSelector>(_tempFile, _simulator, everyN);
        _simulator->run();
        for (int i = 0; i <= 3; i++) {
            std::ostringstream filename;
            filename << _tempFile << "." << std::setfill('0') << std::setw(4)
                     << i << ".ascii";
            if (i % everyN == 0) {
                TS_ASSERT_FILE(filename.str());
            } else {
                TS_ASSERT_NO_FILE(filename.str());
            }
        }
    }


    void testFileOpenError()
    {
        std::string path("/non/existent/path/prefix");
        ASCIIWriter<TestCell<2>, TestValueSelector> *writer = 
            new ASCIIWriter<TestCell<2>, TestValueSelector>(path, _simulator);
        TS_ASSERT_THROWS_ASSERT(
            writer->initialized(),
            FileOpenException &e, 
            TS_ASSERT_SAME_DATA(
                path.c_str(), 
                e.file().c_str(), 
                path.length()));
    }

private:
    std::string _tempFile;
    MonolithicSimulator<TestCell<2> > *_simulator;
};

};
