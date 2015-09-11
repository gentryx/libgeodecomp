#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/parallelization/hpxsimulator.h>

namespace LibGeoDecomp {
namespace HpxSimulator {
namespace HpxSimulatorHelpers {

std::map<std::string, hpx::lcos::local::promise<std::size_t> > localUpdateGroups;
std::map<std::string, hpx::lcos::local::promise<std::size_t> > globalUpdateGroups;
std::map<std::string, hpx::lcos::local::promise<std::vector<std::size_t> > > localityIndices;

std::size_t getNumberOfUpdateGroups(const std::string& basename)
{
    return localUpdateGroups[basename].get_future().get();
}

void setNumberOfUpdateGroups(
    const std::string& basename,
    const std::size_t totalUpdateGroups,
    const std::vector<std::size_t>& indices)
{
    globalUpdateGroups[basename].set_value(totalUpdateGroups);
    localityIndices[basename].set_value(indices);
}

}
}
}

HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulator::HpxSimulatorHelpers::getNumberOfUpdateGroups, getNumberOfUpdateGroups_action);
HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulator::HpxSimulatorHelpers::setNumberOfUpdateGroups, setNumberOfUpdateGroups_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getNumberOfUpdateGroups_action)
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setNumberOfUpdateGroups_action)

HPX_REGISTER_BROADCAST_ACTION(getNumberOfUpdateGroups_action)
HPX_REGISTER_BROADCAST_ACTION(setNumberOfUpdateGroups_action)

namespace LibGeoDecomp {
namespace HpxSimulator {
namespace HpxSimulatorHelpers {

/**
 * Initially we don't have global knowledge on how many
 * UpdateGroups we'll create on each locality. For domain
 * decomposition, we need the sum and also the indices per
 * locality (e.g. given 3 localities with 8, 10, and 2
 * UpdateGroups respectively. Indices per locality: [0, 8, 18])
 */
void gatherAndBroadcastLocalityIndices(
    const std::string& basename,
    const std::vector<double> updateGroupWeights)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    localUpdateGroups[basename].set_value(updateGroupWeights.size());

    if (hpx::get_locality_id() != 0) {
        return;
    }

    std::vector<std::size_t> globalUpdateGroupNumbers =
        hpx::lcos::broadcast<getNumberOfUpdateGroups_action>(
            localities,
            basename).get();

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


}
}
}

#endif
