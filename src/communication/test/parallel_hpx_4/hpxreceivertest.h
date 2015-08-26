#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/misc/stringops.h>

// fixme: find unified macro for this, or even better: define template magic for this, akin to serialization macros
typedef LibGeoDecomp::HPXReceiver<std::vector<int> > ReceiverType1;
typedef LibGeoDecomp::HPXReceiver<hpx::serialization::serialize_buffer<char> > ReceiverType2;

typedef hpx::components::simple_component<ReceiverType1> ReceiverType1Component;
HPX_REGISTER_COMPONENT(ReceiverType1Component, ReceiverType1ComponentVectorInt );
typedef ReceiverType1::receiveAction ReceiverType1ComponentVectorIntReceiveActionReceiveAction;
HPX_REGISTER_ACTION(ReceiverType1ComponentVectorIntReceiveActionReceiveAction);

typedef hpx::components::simple_component<ReceiverType2> ReceiverType2Component;
HPX_REGISTER_COMPONENT(ReceiverType2Component, ReceiverType2ComponentVectorInt );
typedef ReceiverType2::receiveAction ReceiverType2ComponentVectorIntReceiveActionReceiveAction;
HPX_REGISTER_ACTION(ReceiverType2ComponentVectorIntReceiveActionReceiveAction);

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HPXReceiverTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        rank = hpx::get_locality_id();
        size = hpx::find_all_localities().size();

        leftNeighbor  = (rank - 1 + size) % size;
        rightNeighbor = (rank + 1       ) % size;

        name      = "foo/" + StringOps::itoa(rank);
        leftName  = "foo/" + StringOps::itoa(leftNeighbor);
        rightName = "foo/" + StringOps::itoa(rightNeighbor);
    }

    void testBasic()
    {
        boost::shared_ptr<ReceiverType1> receiver = ReceiverType1::make(name).get();

        hpx::id_type leftID  = ReceiverType1::find(leftName ).get();
        hpx::id_type rightID = ReceiverType1::find(rightName).get();

        std::vector<int> sendVec;
        sendVec << rank;
        hpx::apply(ReceiverType1::receiveAction(), leftID,  10, sendVec);
        hpx::apply(ReceiverType1::receiveAction(), rightID, 11, sendVec);

        std::vector<int> fromLeft  = receiver->get(11).get();
        std::vector<int> fromRight = receiver->get(10).get();
        TS_ASSERT_EQUALS(1, fromLeft.size());
        TS_ASSERT_EQUALS(1, fromRight.size());

        TS_ASSERT_EQUALS(leftNeighbor,  fromLeft[0]);
        TS_ASSERT_EQUALS(rightNeighbor, fromRight[0]);

        hpx::unregister_with_basename(name, 0);
    }

    void testWithSerializationBuffer()
    {
        std::vector<char> sendToLeft( leftNeighbor  + 10, char(leftNeighbor));
        std::vector<char> sendToRight(rightNeighbor + 10, char(rightNeighbor));

        
    }

private:
    std::size_t rank;
    std::size_t size;
    std::size_t leftNeighbor;
    std::size_t rightNeighbor;

    std::string name;
    std::string leftName;
    std::string rightName;
};

}
