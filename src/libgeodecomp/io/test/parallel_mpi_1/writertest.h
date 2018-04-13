#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/writer.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <cxxtest/TestSuite.h>
#include <vector>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class HollowWriter : public Writer<CELL_TYPE>
{
public:
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    explicit HollowWriter(
        const std::string& prefix,
        unsigned period = 1) :
        Writer<CELL_TYPE>(prefix, period)
    {}

    Writer<CELL_TYPE> *clone() const
    {
        return 0;
    }

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
};

}
