#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/misc/stringops.h>

typedef LibGeoDecomp::HPXReceiver<std::vector<int> > ReceiverType;

typedef hpx::components::simple_component<ReceiverType> ReceiverTypeComponent;
HPX_REGISTER_COMPONENT(ReceiverTypeComponent, ReceiverTypeComponentVectorInt );
typedef ReceiverType::receiveAction ReceiverTypeComponentVectorIntReceiveActionReceiveAction;
HPX_REGISTER_ACTION(ReceiverTypeComponentVectorIntReceiveActionReceiveAction);

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HPXReceiverTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::size_t rank = hpx::get_locality_id();
        std::size_t size = hpx::find_all_localities().size();

        std::size_t leftNeighbor  = (rank - 1 + size) % size;
        std::size_t rightNeighbor = (rank + 1       ) % size;

        std::string name      = "foo/" + StringOps::itoa(rank);
        std::string leftName  = "foo/" + StringOps::itoa(leftNeighbor);
        std::string rightName = "foo/" + StringOps::itoa(rightNeighbor);

        boost::shared_ptr<ReceiverType> receiver = ReceiverType::make(name).get();

        hpx::id_type leftID  = ReceiverType::find(leftName ).get();
        hpx::id_type rightID = ReceiverType::find(rightName).get();

        std::vector<int> sendVec;
        sendVec << rank;
        hpx::apply(ReceiverType::receiveAction(), leftID,  10, sendVec);
        hpx::apply(ReceiverType::receiveAction(), rightID, 11, sendVec);

        std::vector<int> fromLeft  = receiver->get(11).get();
        std::vector<int> fromRight = receiver->get(10).get();
        TS_ASSERT_EQUALS(1, fromLeft.size());
        TS_ASSERT_EQUALS(1, fromRight.size());

        TS_ASSERT_EQUALS(leftNeighbor,  fromLeft[0]);
        TS_ASSERT_EQUALS(rightNeighbor, fromRight[0]);

        hpx::unregister_id_with_basename(name.c_str(), 0);
    }
};

}
