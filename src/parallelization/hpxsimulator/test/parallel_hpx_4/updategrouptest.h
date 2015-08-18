#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxstepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroup.h>
#include <libgeodecomp/config.h>

namespace LibGeoDecomp {

namespace DummySimulatorHelpers {

hpx::lcos::local::promise<std::size_t> localUpdateGroups;
hpx::lcos::local::promise<std::size_t> globalUpdateGroups;
hpx::lcos::local::promise<std::vector<std::size_t> > localityIndices;

}



/**
 * in HPXSimulator
 * - allgather number of UpdateGroups per locality
 * - set up weight vector
 * - create UpdateGroups
 * - allgather updategroup ids
 * - updateGroups.inits(allIDs)
 *
 * in UpdateGroup.init
 * - allgather bounding boxes
 * - create patchlinks
 * - register patchlinks with basename
 * - let patchlinks look up remote IDs via basenames
 */

template<typename CELL>
struct DummyUpdateGroup : hpx::components::simple_component_base<DummyUpdateGroup<CELL> >
{
    DummyUpdateGroup()
    {
        localID = localIndexCounter++;
        std::cout << "creating DummyUpdateGroup " << localID << "\n";
    }

    hpx::id_type call() const
    {
        std::cout << "DummyUpdateGroup::call()";
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(DummyUpdateGroup, call, call_action);

    static boost::atomic<std::size_t> localIndexCounter;

private:
    std::size_t localID;
};

template<typename CELL>
boost::atomic<std::size_t> DummyUpdateGroup<CELL>::localIndexCounter;

std::size_t getNumberOfUpdateGroups()
{
    hpx::lcos::future<std::size_t> future = LibGeoDecomp::DummySimulatorHelpers::localUpdateGroups.get_future();
    return future.get();
}

void setNumberOfUpdatesGroups(const std::size_t globalUpdateGroups, const std::vector<std::size_t>& indices)
{
    LibGeoDecomp::DummySimulatorHelpers::globalUpdateGroups.set_value(globalUpdateGroups);
    LibGeoDecomp::DummySimulatorHelpers::localityIndices.set_value(indices);
}

}

typedef hpx::components::simple_component<LibGeoDecomp::DummyUpdateGroup<int> > server_type_int;
HPX_REGISTER_COMPONENT(server_type_int, DummyUpdateGroup_int );
typedef hpx::components::simple_component<LibGeoDecomp::DummyUpdateGroup<std::size_t> > server_type_std_size_t;
HPX_REGISTER_COMPONENT(server_type_std_size_t, DummyUpdateGroup_std_size_t );

typedef LibGeoDecomp::DummyUpdateGroup<int>::call_action call_action;
HPX_REGISTER_ACTION(call_action);

HPX_PLAIN_ACTION(LibGeoDecomp::getNumberOfUpdateGroups, getNumberOfUpdateGroups_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getNumberOfUpdateGroups_action)
HPX_REGISTER_BROADCAST_ACTION(getNumberOfUpdateGroups_action)

HPX_PLAIN_ACTION(LibGeoDecomp::setNumberOfUpdatesGroups, setNumberOfUpdatesGroups_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setNumberOfUpdatesGroups_action)
HPX_REGISTER_BROADCAST_ACTION(setNumberOfUpdatesGroups_action)

namespace LibGeoDecomp {

template<typename CELL>
class DummySimulator
{
public:
    DummySimulator(
        std::size_t localUpdateGroups = 10,
        std::string basename = typeid(DummySimulator).name()) :
        localUpdateGroups(localUpdateGroups),
        basename(basename),
        here(hpx::find_here()),
        localities(hpx::find_all_localities())
    {
        DummySimulatorHelpers::localUpdateGroups.set_value(localUpdateGroups);

        if (hpx::get_locality_id() == 0) {
            gatherAndBroadcastLocalityIndices();
        }

        saveLocalityIndices();

        // fixme
        using namespace LibGeoDecomp;
        std::cout << "DummySimulator(@" << hpx::get_locality_id() << ") -> " << localityIndices << "/" << globalUpdateGroups << "\n";

        DummyUpdateGroup<std::size_t>::localIndexCounter = localityIndices[hpx::get_locality_id()];
        localUpdateGroupIDs = hpx::new_<DummyUpdateGroup<std::size_t>[]>(
            hpx::find_here(), localUpdateGroups).get();

        for (std::size_t i = 0; i < localUpdateGroups; ++i) {
            std::size_t id = localityIndices[hpx::get_locality_id()] + i;
            hpx::register_id_with_basename(basename.c_str(), localUpdateGroupIDs[i], id).get();
        }
    }

    ~DummySimulator()
    {
        // fixme: unregister updategroups
    }

    // int getNumberOfUpdateGroups() const
    // {
    //     return oversubscriptionFactor;
    // }


    // HPX_DEFINE_COMPONENT_ACTION(DummySimulator, getNumberOfUpdateGroups, getNumberOfUpdateGroups_action);

private:
    std::size_t localUpdateGroups;
    std::size_t globalUpdateGroups;
    std::vector<std::size_t> localityIndices;
    std::vector<hpx::id_type> localUpdateGroupIDs;
    std::string basename;
    hpx::id_type here;
    std::vector<hpx::id_type> localities;


    /**
     * Initially we don't have global knowledge on how many
     * UpdateGroups we'll create on each locality. For domain
     * decomposition, we need the sum and also the indices per
     * locality (e.g. given 3 localities with 8, 10, and 2
     * UpdateGroups respectively. Indices per locality: [0, 8, 18])
     */
    void gatherAndBroadcastLocalityIndices()
    {
        // fixme
        using namespace LibGeoDecomp;

        std::vector<std::size_t> globalUpdateGroupNumbers =
            hpx::lcos::broadcast<getNumberOfUpdateGroups_action>(localities).get();
        std::vector<std::size_t> indices;
        indices.reserve(globalUpdateGroupNumbers.size());

        std::size_t sum = 0;
        for (auto i: globalUpdateGroupNumbers) {
            indices << sum;
            sum += i;
        }

        hpx::lcos::broadcast<setNumberOfUpdatesGroups_action>(
            localities,
            sum,
            indices).get();
    }

    void saveLocalityIndices()
    {
        globalUpdateGroups = DummySimulatorHelpers::globalUpdateGroups.get_future().get();
        localityIndices = DummySimulatorHelpers::localityIndices.get_future().get();

        // fixme: how to reset?
        // DummySimulatorHelpers::globalUpdateGroups.reset();
        // DummySimulatorHelpers::localityIndices.reset();
    }
};

// td::vector<hpx::id_type> getUpdateGroupIDs()
// {
//     std::cout << "broadcast called\n";
//     for (int i = 0; i < 10; ++i) {
//         localUpdateGroupIDs.push_back(hpx::new_<test_server<int> >(hpx::find_here()).get());
//     }
//     // localUpdateGroupIDs = hpx::new_<test_server<int>[]>(hpx::find_here(), 10).get();
//     return localUpdateGroupIDs;
// }

// HPX_PLAIN_ACTION(getUpdateGroupIDs);
// HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getUpdateGroupIDs_action)
// HPX_REGISTER_BROADCAST_ACTION(getUpdateGroupIDs_action)

// void setUpdateGroupIDs(std::vector<hpx::id_type> ids)
// {
//     std::cout << "setting globalUpdateGroupIDs, size: " << ids.size() << "\n";
//     globalUpdateGroupIDs = ids;
// }

// HPX_PLAIN_ACTION(setUpdateGroupIDs);
// HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setUpdateGroupIDs_action)
// HPX_REGISTER_BROADCAST_ACTION(setUpdateGroupIDs_action)

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

}

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
        DummySimulator<int> sim;
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
