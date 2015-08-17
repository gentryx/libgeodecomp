#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxstepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroup.h>
#include <libgeodecomp/config.h>

std::vector<hpx::id_type> localUpdateGroupIDs;
std::vector<hpx::id_type> globalUpdateGroupIDs;
boost::atomic<std::size_t> localIndexCounter;

template<typename CELL>
struct test_server : hpx::components::simple_component_base<test_server<int> >
{
    test_server()
    {
        localID = localIndexCounter++;
        std::cout << "creating test_server " << localID << "\n";
    }

    hpx::id_type call() const
    {
        std::cout << "test_server::call()";
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);

private:
    std::size_t localID;
};


typedef hpx::components::simple_component<test_server<int> > server_type;
HPX_REGISTER_COMPONENT(server_type, test_server_int );

typedef test_server<int>::call_action call_action;
HPX_REGISTER_ACTION(call_action);

class DummySimulator
{
public:
    DummySimulator()
    {
        int oversubscriptionFactor = 10;
        localUpdateGroupIDs = hpx::new_<test_server<int>[]>(hpx::find_here(), oversubscriptionFactor).get();
        char const* basename = "/HPXSimulatorUpdateGroupFixme/";
        for (int i = 0; i < oversubscriptionFactor; ++i) {
            std::size_t id = hpx::get_locality_id() * oversubscriptionFactor + i;
            hpx::register_id_with_basename(basename, localUpdateGroupIDs[i], id).get();
        }
    }

    std::vector<hpx::id_type> localUpdateGroupIDs;
};

std::vector<hpx::id_type> getUpdateGroupIDs()
{
    std::cout << "broadcast called\n";
    for (int i = 0; i < 10; ++i) {
        localUpdateGroupIDs.push_back(hpx::new_<test_server<int> >(hpx::find_here()).get());
    }
    // localUpdateGroupIDs = hpx::new_<test_server<int>[]>(hpx::find_here(), 10).get();
    return localUpdateGroupIDs;
}

HPX_PLAIN_ACTION(getUpdateGroupIDs);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getUpdateGroupIDs_action)
HPX_REGISTER_BROADCAST_ACTION(getUpdateGroupIDs_action)

void setUpdateGroupIDs(std::vector<hpx::id_type> ids)
{
    std::cout << "setting globalUpdateGroupIDs, size: " << ids.size() << "\n";
    globalUpdateGroupIDs = ids;
}

HPX_PLAIN_ACTION(setUpdateGroupIDs);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setUpdateGroupIDs_action)
HPX_REGISTER_BROADCAST_ACTION(setUpdateGroupIDs_action)

// std::vector<hpx::id_type> 




// int f2(int x)
// {
//     std::cout << "f2(" << x << ")\n";
//     return x * 1000 + hpx::get_locality_id();
// }
// HPX_PLAIN_ACTION(f2);

// HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f2_action)
// HPX_REGISTER_BROADCAST_ACTION(f2_action)


// struct test_client
//   : hpx::components::client_base<test_client, test_server>
// {
//     typedef hpx::components::client_base<test_client, test_server>
//         base_type;

//     test_client() {}
//     test_client(hpx::id_type const& id) : base_type(id) {}
//     test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

//     hpx::id_type call() const { return call_action()(this->get_id()); }
// };

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        typedef HpxSimulator::UpdateGroup<
            TestCell<2>,
            RecursiveBisectionPartition<2>,
            HiParSimulator::HpxStepper<TestCell<2> > > UpdateGroupType;

        UpdateGroupType updateGroup;

        hpx::id_type here = hpx::find_here();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        // if (hpx::get_locality_id() == 0) {
        //     std::vector<std::vector<hpx::id_type> > idTree;
        //     idTree = hpx::lcos::broadcast<getUpdateGroupIDs_action>(localities).get();

        //     std::vector<hpx::id_type> allIDs;
        //     for (auto& i: idTree) {
        //         allIDs.insert(allIDs.end(), i.begin(), i.end());
        //     }

        //     std::cout << here << " -> " << allIDs.size()
        //               << ", " << (hpx::get_locality_id() == 0 ? "HERE" : "NAY")
        //               << ", " << hpx::get_locality_id()
        //               << ", " << hpx::get_locality() << "\n";

        //     hpx::lcos::broadcast<setUpdateGroupIDs_action>(localities, allIDs).get();
        // }


        std::cout << "======================================================1\n";
        DummySimulator sim;
        std::cout << "======================================================2\n";
        char const* basename = "/HPXSimulatorUpdateGroupFixme/";
        std::cout << "======================================================3\n";
        std::vector<hpx::future<hpx::id_type> > all_ids =
            hpx::find_all_ids_from_basename(basename, 10 * localities.size());
        std::cout << "======================================================4 " << all_ids.size() << "\n";



        

        // std::vector<int> f2_res;

        // hpx::id_type here = hpx::find_here();
        // std::cout << "XXXX here: " << here << "\n";

        // f2_res = hpx::lcos::broadcast<f2_action>(localities, 1).get();
        // std::cout << "f2_res: " << f2_res << "\n";

        // std::cout << "BOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMERBOOOMER!\n";

        // char const* basename = "/find_all_ids_from_prefix_test/";

        // test_client t1 = test_client::create(hpx::find_here());
        // hpx::id_type client_id = t1.get_id();

        // std::cout << "check1: " << hpx::naming::invalid_id << ", " << client_id << "\n";

        // // register our component with AGAS
        // std::cout << "register_id_with_basename: " << (hpx::register_id_with_basename(basename, client_id).get()) << "\n";

        // // wait for all localities to register their component

        // std::vector<hpx::future<hpx::id_type> > all_ids = hpx::find_all_ids_from_basename(basename, localities.size());
        // std::cout << "all_ids.size() = " << all_ids.size() << "\n"
        //           << "localities.size() = " << localities.size() << "\n";

        // std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";
}
};

}
