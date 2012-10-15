#include <cxxtest/TestSuite.h>
#include <vector>
#include <stdexcept>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/writer.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class HollowWriter : public Writer<CELL_TYPE>
{
public:
    HollowWriter(const std::string& prefix, 
                 MonolithicSimulator<CELL_TYPE> *sim, 
                 unsigned everyN = 1) : 
        Writer<CELL_TYPE>(prefix, sim, everyN) {}

    virtual void initialized() {}
    virtual void stepFinished() {}
    virtual void allDone() {}
};


class WriterTest : public CxxTest::TestSuite 
{
private:
    MonolithicSimulator<TestCell<2> > *_sim;

public:
    void setUp()
    {
        _sim = new SerialSimulator<TestCell<2> >(new TestInitializer<TestCell<2> >());
    }

    void tearDown()
    {
        delete _sim;
    }

    void testEveryNMustBePositive()
    {
        TS_ASSERT_THROWS(HollowWriter<TestCell<2> >("foobar", _sim, 0), 
                         std::invalid_argument);
    }

    void testPrefixShouldNotBeEmpty()
    {
        TS_ASSERT_THROWS(HollowWriter<TestCell<2> >("", _sim, 1), 
                         std::invalid_argument);
    }
};

}
