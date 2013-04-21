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
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    HollowWriter(const std::string& prefix, 
                 unsigned period = 1) : 
        Writer<CELL_TYPE>(prefix, period) 
    {}

    virtual void stepFinished(const GridType&, unsigned, WriterEvent) 
    {}
};


class WriterTest : public CxxTest::TestSuite 
{
private:
    MonolithicSimulator<TestCell<2> > *sim;

public:
    void setUp()
    {
        sim = new SerialSimulator<TestCell<2> >(new TestInitializer<TestCell<2> >());
    }

    void tearDown()
    {
        delete sim;
    }

    void testPeriodMustBePositive()
    {
        TS_ASSERT_THROWS(HollowWriter<TestCell<2> >("foobar", 0), 
                         std::invalid_argument);
    }

    void testPrefixShouldNotBeEmpty()
    {
        TS_ASSERT_THROWS(HollowWriter<TestCell<2> >("", 1), 
                         std::invalid_argument);
    }
};

}
