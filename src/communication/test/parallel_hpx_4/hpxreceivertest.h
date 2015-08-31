#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/misc/stringops.h>

// fixme: lacks deletion of parentheses
#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE(PARAMS, TEMPLATE)  \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any> * hpx_exported_plugins_list_hpx_factory(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename COMPONENT>                                        \
    class hpx_plugin_exporter_factory;                                  \
                                                                        \
    template<PARAMS>                                                    \
    class hpx_plugin_exporter_factory<TEMPLATE > \
    {                                                                   \
    public:                                                             \
        hpx_plugin_exporter_factory()                                   \
        {                                                               \
            static hpx::util::plugin::concrete_factory< hpx::components::component_factory_base, hpx::components::component_factory<hpx::components::simple_component<TEMPLATE>> > cf; \
            hpx::util::plugin::abstract_factory<hpx::components::component_factory_base>* w = &cf; \
                                                                        \
            std::string actname(typeid(hpx::components::simple_component<TEMPLATE>).name()); \
            boost::algorithm::to_lower(actname);                        \
            hpx_exported_plugins_list_hpx_factory()->insert( std::make_pair(actname, w)); \
        }                                                               \
                                                                        \
        static hpx_plugin_exporter_factory instance;                    \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    hpx_plugin_exporter_factory<TEMPLATE> hpx_plugin_exporter_factory<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any>* hpx_exported_plugins_list_hpx_factory(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename COMPONENT>                                        \
    class init_registry_factory_static;                                 \
                                                                        \
    template<PARAMS>                                                    \
    class init_registry_factory_static<TEMPLATE >                       \
    {                                                                   \
    public:                                                             \
        init_registry_factory_static<TEMPLATE>()                        \
        {                                                               \
            hpx::components::static_factory_load_data_type data = { typeid(hpx::components::simple_component<TEMPLATE>).name(), hpx_exported_plugins_list_hpx_factory }; \
            hpx::components::init_registry_factory(data);               \
        }                                                               \
                                                                        \
        static init_registry_factory_static<TEMPLATE> instance;         \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    init_registry_factory_static<TEMPLATE> init_registry_factory_static<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace components {                                              \
                                                                        \
    template <PARAMS> struct unique_component_name<hpx::components::component_factory<hpx::components::simple_component<TEMPLATE > > > \
    {                                                                   \
        typedef char const* type;                                       \
                                                                        \
        static type call(void)                                          \
        {                                                               \
            return typeid(hpx::components::simple_component<TEMPLATE >).name(); \
        }                                                               \
    };                                                                  \
                                                                        \
    }                                                                   \
    }                                                                   \
                                                                        \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any> * hpx_exported_plugins_list_hpx_registry(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename T>                                                \
    class hpx_plugin_exporter_registry;                                 \
                                                                        \
    template<PARAMS>                                                    \
    class hpx_plugin_exporter_registry<TEMPLATE> \
    {                                                                   \
    public:                                                             \
        hpx_plugin_exporter_registry()                                  \
        {                                                               \
            static hpx::util::plugin::concrete_factory< hpx::components::component_registry_base, hpx::components::component_registry<hpx::components::simple_component<TEMPLATE>, ::hpx::components::factory_check> > cf; \
            hpx::util::plugin::abstract_factory<hpx::components::component_registry_base>* w = &cf; \
            std::string actname(typeid(hpx::components::simple_component<TEMPLATE>).name()); \
            boost::algorithm::to_lower(actname);                        \
            hpx_exported_plugins_list_hpx_registry()->insert( std::make_pair(actname, w)); \
        }                                                               \
                                                                        \
        static hpx_plugin_exporter_registry instance;                   \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    hpx_plugin_exporter_registry<TEMPLATE> hpx_plugin_exporter_registry<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace components {                                              \
                                                                        \
    template <PARAMS>                                                   \
    struct unique_component_name<hpx::components::component_registry<hpx::components::simple_component<TEMPLATE >, ::hpx::components::factory_check> > \
    {                                                                   \
        typedef char const* type;                                       \
        static type call (void)                                         \
        {                                                               \
            return typeid(hpx::components::simple_component<TEMPLATE >).name(); \
        }                                                               \
    };                                                                  \
                                                                        \
    }                                                                   \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace traits {                                                  \
                                                                        \
    template<PARAMS, typename ENABLE>                                   \
    __attribute__((visibility("default")))                              \
    components::component_type component_type_database<CARGO, ENABLE>::get() \
    {                                                                   \
        return value;                                                   \
    }                                                                   \
                                                                        \
    template<PARAMS, typename ENABLE>                                   \
    __attribute__((visibility("default")))                              \
    void component_type_database<CARGO, ENABLE>::set( components::component_type t) \
    {                                                                   \
        value = t;                                                      \
    }                                                                   \
                                                                        \
    }                                                                   \
    };                                                                  \
    //

#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANCE(TYPENAME) \
    template struct hpx::components::component_factory<hpx::components::simple_component<TYPENAME>>; \
    template struct hpx::components::component_registry< hpx::components::simple_component<TYPENAME>, ::hpx::components::factory_check>;


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
