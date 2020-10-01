#include <atomic>
#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/lcos_local/receive_buffer.hpp>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/nesting/hpxstepper.h>
#include <libgeodecomp/parallelization/nesting/hpxupdategroup.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

typedef LibGeoDecomp::TestCell<2> TestCell2;
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell2)

namespace LibGeoDecomp {

namespace DummySimulatorHelpers {

// fix: also split this up into multiple tests to see if this helps with reducing memory footprint

std::map<std::string, hpx::lcos::local::promise<std::size_t> > localUpdateGroups;
std::map<std::string, hpx::lcos::local::promise<std::size_t> > globalUpdateGroups;
std::map<std::string, hpx::lcos::local::promise<std::vector<std::size_t> > > localityIndices;
hpx::lcos::local::promise<bool> allDone;

std::string patchProviderName(const std::string& basename, std::size_t sourceRank, std::size_t targetRank)
{
    return basename + "_PatchProvider_" + StringOps::itoa(sourceRank) + "-" + StringOps::itoa(targetRank);
}

}

template<typename CELL>
class DummyPatchLinkProvider : public hpx::components::simple_component_base<DummyPatchLinkProvider<CELL> >
{
public:
    DummyPatchLinkProvider(std::size_t sourceRank = -1, std::size_t targetRank = -1) :
        sourceRank(sourceRank),
        targetRank(targetRank)
    {}

    void receive(std::size_t step, const CoordBox<2>& box)
    {
        receiveBuffer.store_received(step, CoordBox<2>(box));
    }

    HPX_DEFINE_COMPONENT_ACTION(DummyPatchLinkProvider, receive, receive_action);

    void get(std::size_t step)
    {
        receiveBuffer.receive(step).get();
    }

    HPX_DEFINE_COMPONENT_ACTION(DummyPatchLinkProvider, get, get_action);

private:
    hpx::lcos::local::receive_buffer<CoordBox<2> > receiveBuffer;
    std::size_t sourceRank;
    std::size_t targetRank;
};

template<typename CELL>
class DummyPatchLinkAccepter : public hpx::components::simple_component_base<DummyPatchLinkAccepter<CELL> >
{
public:
    DummyPatchLinkAccepter(const std::string& basename = "", std::size_t sourceRank = -1, std::size_t targetRank = -1) :
        basename(basename),
        sourceRank(sourceRank),
        targetRank(targetRank)
    {
        std::string name = DummySimulatorHelpers::patchProviderName(basename, sourceRank, targetRank);

        std::vector<hpx::future<hpx::id_type> > ids = hpx::find_all_from_basename(name, 1);
        if (ids.size() != 1) {
            throw std::logic_error("unexpected amount of PatchProviders found in AGAS, expected exactly 1");
        }
        remoteID = ids[0].get();
    }

    void put(std::size_t step, const CoordBox<2>& box)
    {
        typename DummyPatchLinkProvider<CELL>::receive_action receiveAction;
        hpx::async(receiveAction, remoteID, step, box);
    }

private:
    std::string basename;
    std::size_t sourceRank;
    std::size_t targetRank;
    hpx::id_type remoteID;
};

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
class DummyUpdateGroup : public hpx::components::simple_component_base<DummyUpdateGroup<CELL> >
{
public:
    DummyUpdateGroup(
        std::string basename = "",
        std::size_t globalUpdateGroups = 0,
        const CoordBox<2>& gridDim = CoordBox<2>()) :
        stepCounter(0),
        gridDim(gridDim)
    {
        id = localIndexCounter++;

        std::size_t leftNeighbor  = (id - 1 + globalUpdateGroups) % globalUpdateGroups;
        std::size_t rightNeighbor = (id + 1) % globalUpdateGroups;

        // create PatchProviders
        patchProviders[leftNeighbor ] = hpx::new_<DummyPatchLinkProvider<CELL> >(
            hpx::find_here(), leftNeighbor,  id).get();
        patchProviders[rightNeighbor] = hpx::new_<DummyPatchLinkProvider<CELL> >(
            hpx::find_here(), rightNeighbor, id).get();

        for (auto&& i: patchProviders) {
            std::string name = DummySimulatorHelpers::patchProviderName(basename, id, i.first);
            hpx::register_with_basename(name, i.second, 0).get();
        }

        // create PatchAccepters
        patchAccepters << DummyPatchLinkAccepter<CELL>(basename, id, leftNeighbor);
        patchAccepters << DummyPatchLinkAccepter<CELL>(basename, id, rightNeighbor);
    }

    void step()
    {
        std::vector<hpx::lcos::future<void> > ghostFutures;
        ghostFutures.reserve(patchAccepters.size());

        for (std::size_t i = 0; i < patchAccepters.size(); ++i) {
            ghostFutures << hpx::async(
                &DummyPatchLinkAccepter<CELL>::put, patchAccepters[i], stepCounter + 1, gridDim);
        }

        // would update interior here

        stepCounter += 1;
        for (auto&& i: ghostFutures) {
            i.get();
        }
        std::map<std::size_t, hpx::lcos::future<void> > ghostFutures2;
        typename DummyPatchLinkProvider<CELL>::get_action getAction;
        for (auto&& i: patchProviders) {
            ghostFutures2[i.first] = hpx::async(getAction, i.second, stepCounter);
        }
        for (auto i = ghostFutures2.begin(); i != ghostFutures2.end(); ++i) {
            i->second.get();
        }
    }

    HPX_DEFINE_COMPONENT_ACTION(DummyUpdateGroup, step, step_action);

    static std::atomic<std::size_t> localIndexCounter;

private:
    std::size_t stepCounter;
    std::size_t id;
    std::map<std::size_t, hpx::id_type> patchProviders;
    std::vector<DummyPatchLinkAccepter<CELL> > patchAccepters;
    CoordBox<2> gridDim;
};

template<typename CELL>
std::atomic<std::size_t> DummyUpdateGroup<CELL>::localIndexCounter;

std::size_t getNumberOfUpdateGroups(const std::string& basename)
{
    hpx::lcos::future<std::size_t> future = DummySimulatorHelpers::localUpdateGroups[basename].get_future();
    return future.get();
}

void setNumberOfUpdateGroups(
    const std::string& basename,
    const std::size_t globalUpdateGroups,
    const std::vector<std::size_t>& indices)
{
    DummySimulatorHelpers::globalUpdateGroups[basename].set_value(globalUpdateGroups);
    DummySimulatorHelpers::localityIndices[basename].set_value(indices);
}

void allDone()
{
    DummySimulatorHelpers::allDone.get_future().get();
}

}

// register component
typedef hpx::components::simple_component<LibGeoDecomp::DummyPatchLinkAccepter<int> > DummyPatchLinkAccepterType_int;
HPX_REGISTER_COMPONENT(DummyPatchLinkAccepterType_int, DummyPatchLinkAccepter_int );
typedef hpx::components::simple_component<LibGeoDecomp::DummyPatchLinkAccepter<std::size_t> > DummyPatchLinkAccepterType_stdSizeT;
HPX_REGISTER_COMPONENT(DummyPatchLinkAccepterType_stdSizeT, DummyPatchLinkAccepter_stdSizeT );

// register component
typedef hpx::components::simple_component<LibGeoDecomp::DummyPatchLinkProvider<int> > DummyPatchLinkProviderType_int;
HPX_REGISTER_COMPONENT(DummyPatchLinkProviderType_int, DummyPatchLinkProvider_int );
typedef hpx::components::simple_component<LibGeoDecomp::DummyPatchLinkProvider<std::size_t> > DummyPatchLinkProviderType_stdSizeT;
HPX_REGISTER_COMPONENT(DummyPatchLinkProviderType_stdSizeT, DummyPatchLinkProvider_stdSizeT );

// register action
typedef LibGeoDecomp::DummyPatchLinkProvider<int>::receive_action DummyPatchLinkProvider_receive_action_int;
HPX_REGISTER_ACTION(DummyPatchLinkProvider_receive_action_int);
typedef LibGeoDecomp::DummyPatchLinkProvider<std::size_t>::receive_action DummyPatchLinkProvider_receive_action_stdSizeT;
HPX_REGISTER_ACTION(DummyPatchLinkProvider_receive_action_stdSizeT);

// register component
typedef hpx::components::simple_component<LibGeoDecomp::DummyUpdateGroup<int> > DummyUpdateGroupType_int;
HPX_REGISTER_COMPONENT(DummyUpdateGroupType_int, DummyUpdateGroup_int );
typedef hpx::components::simple_component<LibGeoDecomp::DummyUpdateGroup<std::size_t> > DummyUpdateGroupType_stdSizeT;
HPX_REGISTER_COMPONENT(DummyUpdateGroupType_stdSizeT, DummyUpdateGroup_stdSizeT );

// register action
typedef LibGeoDecomp::DummyUpdateGroup<int>::step_action step_action_int;
HPX_REGISTER_ACTION(step_action_int);
typedef LibGeoDecomp::DummyUpdateGroup<std::size_t>::step_action step_action_stdSizeT;
HPX_REGISTER_ACTION(step_action_stdSizeT);

// register broadcasts
HPX_PLAIN_ACTION(LibGeoDecomp::getNumberOfUpdateGroups, getNumberOfUpdateGroups_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getNumberOfUpdateGroups_action)
HPX_REGISTER_BROADCAST_ACTION(getNumberOfUpdateGroups_action)

HPX_PLAIN_ACTION(LibGeoDecomp::setNumberOfUpdateGroups, setNumberOfUpdateGroups_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setNumberOfUpdateGroups_action)
HPX_REGISTER_BROADCAST_ACTION(setNumberOfUpdateGroups_action)

HPX_PLAIN_ACTION(LibGeoDecomp::allDone, allDone_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(allDone_action)
HPX_REGISTER_BROADCAST_ACTION(allDone_action)

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
        DummySimulatorHelpers::localUpdateGroups[basename].set_value(localUpdateGroups);

        if (hpx::get_locality_id() == 0) {
            gatherAndBroadcastLocalityIndices(basename);
        }

        saveLocalityIndices();

        DummyUpdateGroup<CELL>::localIndexCounter = localityIndices[hpx::get_locality_id()];

        std::vector<hpx::future<hpx::id_type> > tempFutures;
        for (std::size_t i = 0; i < localUpdateGroups; ++i) {
            tempFutures << hpx::new_<DummyUpdateGroup<CELL> >(
                hpx::find_here(),
                basename,
                globalUpdateGroups,
                CoordBox<2>(Coord<2>(100, 200), Coord<2>(300, 200)));
        }
        for (std::size_t i = 0; i < localUpdateGroups; ++i) {
            localUpdateGroupIDs << tempFutures[i].get();
        }

        for (std::size_t i = 0; i < localUpdateGroups; ++i) {
            std::size_t id = localityIndices[hpx::get_locality_id()] + i;
            hpx::register_with_basename(basename, localUpdateGroupIDs[i], id).get();
        }
    }

    ~DummySimulator()
    {
        for (std::size_t i = 0; i < localUpdateGroups; ++i) {
            std::size_t id = localityIndices[hpx::get_locality_id()] + i;
            hpx::unregister_with_basename(basename, id);
        }
    }

    void step()
    {
        typename DummyUpdateGroup<CELL>::step_action stepAction;

        std::vector<hpx::future<void> > updateFutures;
        for (auto&& i: localUpdateGroupIDs) {
            updateFutures << hpx::async(stepAction, i);
        }
        for (auto&& i: updateFutures) {
            i.get();
        }
    }

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
    void gatherAndBroadcastLocalityIndices(const std::string& basename)
    {
        std::vector<std::size_t> globalUpdateGroupNumbers =
            hpx::lcos::broadcast<getNumberOfUpdateGroups_action>(localities, basename).get();
        std::vector<std::size_t> indices;
        indices.reserve(globalUpdateGroupNumbers.size());

        std::size_t sum = 0;
        for (auto&& i: globalUpdateGroupNumbers) {
            indices << sum;
            sum += i;
        }

        hpx::lcos::broadcast<setNumberOfUpdateGroups_action>(
            localities,
            basename,
            sum,
            indices).get();
    }

    void saveLocalityIndices()
    {
        globalUpdateGroups = DummySimulatorHelpers::globalUpdateGroups[basename].get_future().get();
        localityIndices = DummySimulatorHelpers::localityIndices[basename].get_future().get();

        DummySimulatorHelpers::globalUpdateGroups[basename] = hpx::lcos::local::promise<std::size_t>();
        DummySimulatorHelpers::localUpdateGroups[basename] = hpx::lcos::local::promise<std::size_t>();
        DummySimulatorHelpers::localityIndices[basename] = hpx::lcos::local::promise<std::vector<std::size_t> >();
    }
};

}

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:

    typedef RecursiveBisectionPartition<2> PartitionType;
    typedef LibGeoDecomp::HPXStepper<TestCell<2>, UpdateFunctorHelpers::ConcurrencyEnableHPX> StepperType;
    typedef HPXUpdateGroup<TestCell<2> > UpdateGroupType;
    typedef StepperType::GridType GridType;


    void testBasic()
    {
        hpx::id_type here = hpx::find_here();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        DummySimulator<int> sim;
        sim.step();
        sim.step();
        sim.step();
        sim.step();
        sim.step();
        sim.step();

        DummySimulatorHelpers::allDone.set_value(true);
        hpx::lcos::broadcast<allDone_action>(
            localities).get();
    }

    void testCreation()
    {
        SharedPtr<UpdateGroupType>::Type updateGroup;

        rank = hpx::get_locality_id();
        dimensions = Coord<2>(231, 350);
        weights = genWeights(dimensions.x(), dimensions.y(), hpx::get_num_localities().get());
        partition.reset(new PartitionType(Coord<2>(), dimensions, 0, weights));
        ghostZoneWidth = 9;
        init.reset(new TestInitializer<TestCell<2> >(dimensions));
        updateGroup.reset(
            new UpdateGroupType(
                partition,
                CoordBox<2>(Coord<2>(), dimensions),
                ghostZoneWidth,
                init,
                reinterpret_cast<StepperType*>(0)));
        expectedNanoSteps.clear();
        expectedNanoSteps << 5 << 7 << 8 << 33 << 55;
        mockPatchAccepter.reset(new MockPatchAccepter<GridType>());
        for (std::deque<std::size_t>::iterator i = expectedNanoSteps.begin();
             i != expectedNanoSteps.end();
             ++i) {
            mockPatchAccepter->pushRequest(*i);
        }
        updateGroup->addPatchAccepter(mockPatchAccepter, StepperType::INNER_SET);

        updateGroup->update(100);

        std::deque<std::size_t> actualNanoSteps = mockPatchAccepter->getOfferedNanoSteps();
        TS_ASSERT_EQUALS(actualNanoSteps, expectedNanoSteps);
    }

private:
    std::deque<std::size_t> expectedNanoSteps;
    int rank;
    Coord<2> dimensions;
    std::vector<std::size_t> weights;
    unsigned ghostZoneWidth;
    SharedPtr<PartitionType>::Type partition;
    SharedPtr<Initializer<TestCell<2> > >::Type init;
    SharedPtr<UpdateGroupType>::Type updateGroup;
    SharedPtr<MockPatchAccepter<GridType> >::Type mockPatchAccepter;

    std::vector<std::size_t> genWeights(
        unsigned width,
        unsigned height,
        unsigned size)
    {
        std::vector<std::size_t> ret(size);
        unsigned totalSize = width * height;
        for (std::size_t i = 0; i < ret.size(); ++i) {
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        }

        return ret;
    }

    long pos(unsigned i, unsigned size, unsigned totalSize)
    {
        return i * totalSize / size;
    }

};

}
