#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class MockPatchAccepter : public PatchAccepter<GRID_TYPE>
{
public:
    friend class VanillaStepperBasicTest;

    virtual void put(
        const GRID_TYPE& /*grid*/, 
        const Region<2>& /*validRegion*/, 
        const unsigned& nanoStep) 
    {
        offeredNanoSteps.push_back(nanoStep);
        requestedNanoSteps.pop_front();
    }
    
    virtual long nextRequiredNanoStep()
    {
        return requestedNanoSteps.front();
    }

    void pushRequest(const long& nanoStep)
    {
        requestedNanoSteps.push_back(nanoStep);
    }

    const std::deque<long>& getRequestedNanoSteps() const
    {
        return requestedNanoSteps;
    }

    const std::deque<long>& getOfferedNanoSteps() const
    {
        return offeredNanoSteps;
    }

private:
    std::deque<long> requestedNanoSteps;
    std::deque<long> offeredNanoSteps;
};

class VanillaStepperBasicTest : public CxxTest::TestSuite
{
public:
    typedef DisplacedGrid<TestCell<2>, Topologies::Cube<2>::Topology, true> GridType;

    void setUp()
    {
        init.reset(new TestInitializer<2>(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();

        patchAccepter.reset(new MockPatchAccepter<GridType>());
        patchAccepter->pushRequest(2);
        patchAccepter->pushRequest(10);
        patchAccepter->pushRequest(13);
   
        partitionManager.reset(new PartitionManager<2>(rect));
        stepper.reset(
            new VanillaStepper<TestCell<2>, 2>(partitionManager, init));

        stepper->addPatchAccepter(patchAccepter);
    }

    void testUpdate()
    {
        // fixme:
    //     TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 0);
    //     stepper->update();
    //     TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 1);
    // }

    // void testUpdateMultiple()
    // {
    //     stepper->update(8);
    //     TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 8);
    //     stepper->update(30);
    //     TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 38);
    // }

    // void testPutPatch()
    // {
    //     stepper->update(9);
    //     TS_ASSERT_EQUALS(1, patchAccepter->offeredNanoSteps.size());
    //     stepper->update(4);
    //     TS_ASSERT_EQUALS(3, patchAccepter->offeredNanoSteps.size());
    }

private:
    boost::shared_ptr<TestInitializer<2> > init;
    boost::shared_ptr<PartitionManager<2> > partitionManager;
    boost::shared_ptr<VanillaStepper<TestCell<2>, 2> > stepper;
    boost::shared_ptr<MockPatchAccepter<GridType> > patchAccepter;
};

}
}
