#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_CREATEUPDATEGROUPS_H
#define LIBGEODECOMP_PARALLELIZATION_CREATEUPDATEGROUPS_H

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/util/locality_result.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <libgeodecomp/io/initializer.h>

#include <utility>
#include <vector>

namespace LibGeoDecomp {
namespace HpxSimulator {
namespace Implementation {

typedef
    std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
    CreateUpdateGroupsReturnType;

std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
createUpdateGroups(std::vector<hpx::id_type> localities, hpx::components::component_type type, float overcommitFactor);

HPX_DEFINE_PLAIN_ACTION(createUpdateGroups, CreateUpdateGroupsAction);

} // namespace Implementation

template <class UPDATEGROUP>
inline std::vector<UPDATEGROUP> createUpdateGroups(
    float overcommitFactor
)
{
    hpx::components::component_type type =
        hpx::components::get_component_type<typename UPDATEGROUP::ComponentType>();

    std::vector<hpx::id_type> localities = hpx::find_all_localities(type);

    hpx::id_type id = localities[0];
    hpx::future<std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> > >
        asyncResult = hpx::async<Implementation::CreateUpdateGroupsAction>(
            id, boost::move(localities), type, overcommitFactor);

    std::vector<UPDATEGROUP> components;

    std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
        result(boost::move(asyncResult.move()));

    std::size_t numComponents = result.first;
    components.reserve(numComponents);

    std::vector<hpx::util::locality_result> res;
    res.reserve(result.second.size());
    BOOST_FOREACH(const hpx::util::remote_locality_result& rl, result.second) {
        res.push_back(rl);
    }

    BOOST_FOREACH(hpx::id_type id, hpx::util::locality_results(res)) {
        components.push_back(id);
    }

    return components;
}

}
}

HPX_REGISTER_PLAIN_ACTION_DECLARATION(
    LibGeoDecomp::HpxSimulator::Implementation::CreateUpdateGroupsAction
)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    LibGeoDecomp::HpxSimulator::Implementation::CreateUpdateGroupsReturnType,
    hpx_base_lco_std_pair_std_size_t_std_vector_hpx_util_remote_locality_result
)

#endif
#endif
