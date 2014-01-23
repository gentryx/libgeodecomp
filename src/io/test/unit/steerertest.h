#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

#include <boost/shared_ptr.hpp>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyTestCell
{
public:
    class API : public APITraits::HasStaticData<int>
    {};

    static int staticData;

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD hood, unsigned nanoStep)
    {
        // NOP
    }
};

int MyTestCell::staticData = 0;

class MyStaticDataModifyingSteerer : public Steerer<MyTestCell>
{
public:
    MyStaticDataModifyingSteerer() :
        Steerer<MyTestCell>(1),
        counter(666)
    {}

    virtual void nextStep(
        GridType *grid,
        const Region<Topology::DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        feedback->setStaticData(counter++);
    }

private:
    int counter;
};

class MyInitializer : public SimpleInitializer<MyTestCell>
{
public:
    MyInitializer(Coord<2> dim, unsigned maxSteps) :
        SimpleInitializer<MyTestCell>(dim, maxSteps)
    {}

    virtual void grid(GridBase<MyTestCell, 2> *target)
    {}
};

class SteererTest : public CxxTest::TestSuite
{
public:
    typedef MyStaticDataModifyingSteerer SteererType;

    void setUp()
    {
        simulator.reset(
            new SerialSimulator<MyTestCell>(
                new MyInitializer(
                    Coord<2>(10, 20),
                    34)));
        simulator->addSteerer(new SteererType());
    }

    void tearDown()
    {
        simulator.reset();
    }

    void testStaticDataModification()
    {
        TS_ASSERT_EQUALS(0, MyTestCell::staticData);
        simulator->run();
        // initial value + 34 steps + finsh:
        TS_ASSERT_EQUALS(666 + 34 + 1, MyTestCell::staticData);
    }

private:
    boost::shared_ptr<SerialSimulator<MyTestCell> > simulator;
};

}
