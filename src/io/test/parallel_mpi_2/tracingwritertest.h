#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>

#include <sstream>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TracingWriterTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        simulator = new StripingSimulator<TestCell<2> >(
            new TestInitializer<TestCell<2> >(),
            MPILayer().rank() ? 0 : new NoOpBalancer(),
            1000);
    }

    void tearDown()
    {
        delete simulator;
    }

    void testOutputToStream()
    {
        std::ostringstream output;
        simulator->addWriter(
            new TracingWriter<TestCell<2> >(
                1,
                simulator->getInitializer()->maxSteps(),
                -1,
                output));

        simulator->run();

        // collect some substrings we expect the output to contain
        std::vector<std::string> s;
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
    DistributedSimulator<TestCell<2> > *simulator;
};

}
