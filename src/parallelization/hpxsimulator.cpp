#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/parallelization/hpxsimulator.h>

namespace LibGeoDecomp {
namespace HpxSimulatorHelpers {

std::map<std::string, hpx::lcos::local::promise<std::vector<double> > > localUpdateGroupSpeeds;
std::map<std::string, hpx::lcos::local::promise<std::vector<double> > > myGlobalUpdateGroupSpeeds;
std::map<std::string, hpx::lcos::local::promise<std::vector<std::size_t> > > myLocalityIndices;

std::vector<double> getUpdateGroupSpeeds(const std::string& basename)
{
    return localUpdateGroupSpeeds[basename].get_future().get();
}

void setNumberOfUpdateGroups(
    const std::string& basename,
    const std::vector<double>& updateGroupSpeeds,
    const std::vector<std::size_t>& indices)
{
    myGlobalUpdateGroupSpeeds[basename].set_value(updateGroupSpeeds);
    myLocalityIndices[basename].set_value(indices);
}

}
}

HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulatorHelpers::getUpdateGroupSpeeds, getUpdateGroupSpeeds_action);
HPX_PLAIN_ACTION(LibGeoDecomp::HpxSimulatorHelpers::setNumberOfUpdateGroups, setNumberOfUpdateGroups_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(getUpdateGroupSpeeds_action)
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(setNumberOfUpdateGroups_action)

HPX_REGISTER_BROADCAST_ACTION(getUpdateGroupSpeeds_action)
HPX_REGISTER_BROADCAST_ACTION(setNumberOfUpdateGroups_action)

namespace LibGeoDecomp {
namespace HpxSimulatorHelpers {

/**
 * Initially we don't have global knowledge on how many
 * UpdateGroups we'll create on each locality. For domain
 * decomposition, we need the sum and also the indices per
 * locality (e.g. given 3 localities with 8, 10, and 2
 * UpdateGroups respectively. Indices per locality: [0, 8, 18])
 */
void gatherAndBroadcastLocalityIndices(
    double speedGuide,
    std::vector<double> *globalUpdateGroupSpeeds,
    std::vector<std::size_t> *localityIndices,
    const std::string& basename,
    const std::vector<double> updateGroupSpeeds)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<double> correctedUpdateGroupSpeeds;
    correctedUpdateGroupSpeeds.reserve(updateGroupSpeeds.size());
    for (double i: updateGroupSpeeds) {
        correctedUpdateGroupSpeeds << i * speedGuide;
    }

    localUpdateGroupSpeeds[basename].set_value(correctedUpdateGroupSpeeds);

    if (hpx::get_locality_id() == 0) {
        std::vector<std::vector<double> > tempGlobalUpdateGroupSpeeds =
            hpx::lcos::broadcast<getUpdateGroupSpeeds_action>(localities, basename).get();

        std::vector<std::size_t> indices;
        std::vector<double> flattenedUpdateGroupSpeeds;
        std::size_t indexSum = 0;

        for (auto&& vec: tempGlobalUpdateGroupSpeeds) {
            for (auto&& weight: vec) {
                flattenedUpdateGroupSpeeds << weight;
            }

            indices << indexSum;
            indexSum += vec.size();
        }
        indices << indexSum;

        hpx::lcos::broadcast<setNumberOfUpdateGroups_action>(
            localities,
            basename,
            flattenedUpdateGroupSpeeds,
            indices).get();
    }

    *globalUpdateGroupSpeeds = myGlobalUpdateGroupSpeeds[basename].get_future().get();
    *localityIndices = myLocalityIndices[basename].get_future().get();
}


}
}

#endif
