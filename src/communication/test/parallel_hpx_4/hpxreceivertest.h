#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/misc/stringops.h>


LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE(typename CARGO, LibGeoDecomp::HPXReceiver<CARGO>)


typedef LibGeoDecomp::HPXReceiver<std::vector<int> > ReceiverType1;
typedef LibGeoDecomp::HPXReceiver<hpx::serialization::serialize_buffer<char> > ReceiverType2;
typedef LibGeoDecomp::HPXReceiver<double> ReceiverType3;

LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANCE(ReceiverType1);
LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANCE(ReceiverType2);
LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANCE(ReceiverType3);

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HPXReceiverTest : public CxxTest::TestSuite
{
public:
    void mySetUp(std::string basename)
    {
        rank = hpx::get_locality_id();
        size = hpx::find_all_localities().size();

        leftNeighbor  = (rank - 1 + size) % size;
        rightNeighbor = (rank + 1       ) % size;

        name      = basename + "/" + StringOps::itoa(rank);
        leftName  = basename + "/" + StringOps::itoa(leftNeighbor);
        rightName = basename + "/" + StringOps::itoa(rightNeighbor);
    }

    void testBasic()
    {
        mySetUp("HPXReceiverTest::testBasic");
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
        mySetUp("HPXReceiverTest::testWithSerializationBuffer");

        std::vector<char> sendToLeft( rank + 10, char(leftNeighbor));
        std::vector<char> sendToRight(rank + 10, char(rightNeighbor));
        hpx::serialization::serialize_buffer<char> sendLeftBuffer( &sendToLeft[0],  sendToLeft.size());
        hpx::serialization::serialize_buffer<char> sendRightBuffer(&sendToRight[0], sendToRight.size());

        boost::shared_ptr<ReceiverType2> receiver = ReceiverType2::make(name).get();

        hpx::id_type leftID  = ReceiverType2::find(leftName ).get();
        hpx::id_type rightID = ReceiverType2::find(rightName).get();

        hpx::apply(ReceiverType2::receiveAction(), leftID,  20, sendLeftBuffer);
        hpx::apply(ReceiverType2::receiveAction(), rightID, 21, sendRightBuffer);

        hpx::serialization::serialize_buffer<char> fromLeft  = receiver->get(21).get();
        hpx::serialization::serialize_buffer<char> fromRight = receiver->get(20).get();

        TS_ASSERT_EQUALS(rank, std::size_t(fromLeft[0]));
        TS_ASSERT_EQUALS(rank, std::size_t(fromRight[0]));
        TS_ASSERT_EQUALS(10 + leftNeighbor,  fromLeft.size());
        TS_ASSERT_EQUALS(10 + rightNeighbor, fromRight.size());

        hpx::unregister_with_basename(name, 0);
    }

    void testWithDouble()
    {
        mySetUp("HPXReceiverTest::testWithDouble");

        double sendToLeft  = rank + 10.123;
        double sendToRight = rank + 10.234;

        boost::shared_ptr<ReceiverType3> receiver = ReceiverType3::make(name).get();

        hpx::id_type leftID  = ReceiverType3::find(leftName ).get();
        hpx::id_type rightID = ReceiverType3::find(rightName).get();

        hpx::apply(ReceiverType3::receiveAction(), leftID,  30, sendToLeft);
        hpx::apply(ReceiverType3::receiveAction(), rightID, 31, sendToRight);

        double fromLeft  = receiver->get(31).get();
        double fromRight = receiver->get(30).get();

        TS_ASSERT_EQUALS(10.234 + leftNeighbor,  fromLeft);
        TS_ASSERT_EQUALS(10.123 + rightNeighbor, fromRight);

        hpx::unregister_with_basename(name, 0);
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
