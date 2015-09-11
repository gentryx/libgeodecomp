#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/parallelization/hpxsimulator.h>

namespace LibGeoDecomp {
namespace HpxSimulator {
namespace HpxSimulatorHelpers {

std::map<std::string, hpx::lcos::local::promise<std::vector<double> > > localUpdateGroupWeights;
std::map<std::string, hpx::lcos::local::promise<std::vector<double> > > globalUpdateGroupWeights;
std::map<std::string, hpx::lcos::local::promise<std::vector<std::size_t> > > localityIndices;

std::vector<double> getUpdateGroupWeights(const std::string& basename)
{
    return localUpdateGroupWeights[basename].get_future().get();
}

void setNumberOfUpdateGroups(
    const std::string& basename,
    const std::vector<double>& updateGroupWeights,
    const std::vector<std::size_t>& indices)
{
    globalUpdateGroupWeights[basename].set_value(updateGroupWeights);
    localityIndices[basename].set_value(indices);
}

}
}
}

HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulator::HpxSimulatorHelpers::getUpdateGroupWeights, getUpdateGroupWeights_action);
HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulator::HpxSimulatorHelpers::setNumberOfUpdateGroups, setNumberOfUpdateGroups_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getUpdateGroupWeights_action)
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setNumberOfUpdateGroups_action)

HPX_REGISTER_BROADCAST_ACTION(getUpdateGroupWeights_action)
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

    localUpdateGroupWeights[basename].set_value(updateGroupWeights);

    if (hpx::get_locality_id() != 0) {
        return;
    }

    std::vector<std::vector<double> > globalUpdateGroupWeights =
        hpx::lcos::broadcast<getUpdateGroupWeights_action>(localities, basename).get();

    std::vector<std::size_t> indices;
    std::vector<double> flattenedUpdateGroupWeights;
    std::size_t indexSum = 0;

    for (auto&& vec: globalUpdateGroupWeights) {
        for (auto&& weight: vec) {
            flattenedUpdateGroupWeights << weight;
        }

        indices << indexSum;
        indexSum += vec.size();
    }

    hpx::lcos::broadcast<setNumberOfUpdateGroups_action>(
        localities,
        basename,
        flattenedUpdateGroupWeights,
        indices).get();
}


}
}
}

#endif
