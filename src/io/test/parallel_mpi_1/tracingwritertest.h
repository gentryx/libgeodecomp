#include <sstream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TracingWriterTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        simulator = new SerialSimulator<TestCell<2> >(
            new TestInitializer<TestCell<2> >());
    }


    void tearDown() {
        delete simulator;
    }


    void testOutputToStream()
    {
        std::ostringstream output;
        simulator->addWriter( 
            new TracingWriter<TestCell<2> >(
                1, TestInitializer<TestCell<2> >().maxSteps(), output));
        simulator->run();

        // collect some substrings we expect the output to contain
        SuperVector<std::string> s;
        s.push_back("TracingWriter::initialized()");
        s.push_back("  time");
        s.push_back("TracingWriter::stepFinished()");
        s.push_back("  step");
        s.push_back("  ETA");
        s.push_back("  time");
        s.push_back("TracingWriter::allDone()");
        s.push_back("  total time");
        s.push_back("  time");

        long unsigned index = 0;
        for (unsigned i = 0; i < s.size(); i++) {
            index = output.str().find(s[i], index);
            TS_ASSERT(index != std::string::npos);
        }
    }

private:
    MonolithicSimulator<TestCell<2> > *simulator;
};

}
