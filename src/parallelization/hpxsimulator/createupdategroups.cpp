#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX

#include <libgeodecomp/parallelization/hpxsimulator/createupdategroups.h>

#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/wait_any.hpp>

#include <boost/serialization/vector.hpp>

typedef LibGeoDecomp::HpxSimulator::Impl::CreateUpdateGroupsAction CreateUpdateGroupsAction;

HPX_REGISTER_PLAIN_ACTION(
    LibGeoDecomp::HpxSimulator::Impl::CreateUpdateGroupsAction,
    LibGeoDecomp_HpxSimulator_Impl_CreateUpdateGroupsAction
);

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::HpxSimulator::Impl::CreateUpdateGroupsReturnType,
    hpx_base_lco_std_pair_std_size_t_std_vector_hpx_util_remote_locality_result
)

namespace LibGeoDecomp {
namespace HpxSimulator {
namespace Impl {

std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
createUpdateGroups(std::vector<hpx::id_type> localities, hpx::components::component_type type, std::size_t overcommitFactor)
{
    typedef hpx::util::remote_locality_result ValueType;
    typedef std::pair<std::size_t, std::vector<ValueType> > ResultType;

    ResultType res;
    if(localities.size() == 0) return res;

    hpx::id_type thisLoc = localities[0];

    typedef
        hpx::components::server::runtime_support::bulk_create_components_action
        ActionType;

    std::size_t numComponents = hpx::get_os_thread_count() * overcommitFactor;

    typedef hpx::future<std::vector<hpx::naming::gid_type> > FutureType;

    FutureType f;
    {
        hpx::lcos::packaged_action<ActionType, std::vector<hpx::naming::gid_type> > p;
        p.apply(hpx::launch::async, thisLoc, type, numComponents);
        f = p.get_future();
    }

    std::vector<hpx::future<ResultType> > componentsFutures;
    componentsFutures.reserve(2);

    if(localities.size() > 1)
    {
        std::size_t half = (localities.size() / 2) + 1;
        std::vector<hpx::id_type> locsFirst(localities.begin() + 1, localities.begin() + half);
        std::vector<hpx::id_type> locsSecond(localities.begin() + half, localities.end());


        if(locsFirst.size() > 0)
        {
            hpx::lcos::packaged_action<CreateUpdateGroupsAction, ResultType > p;
            hpx::id_type id = locsFirst[0];
            p.apply(hpx::launch::async, id, boost::move(locsFirst), type, overcommitFactor);
            componentsFutures.push_back(
                p.get_future()
            );
        }

        if(locsSecond.size() > 0)
        {
            hpx::lcos::packaged_action<CreateUpdateGroupsAction, ResultType > p;
            hpx::id_type id = locsSecond[0];
            p.apply(hpx::launch::async, id, boost::move(locsSecond), type, overcommitFactor);
            componentsFutures.push_back(
                p.get_future()
            );
        }
    }

    res.first = numComponents;
    res.second.push_back(
        ValueType(thisLoc.get_gid(), type)
    );
    res.second.back().gids_ = boost::move(f.move());

    while(!componentsFutures.empty())
    {
        HPX_STD_TUPLE<int, hpx::future<ResultType> >
            compRes = hpx::wait_any(componentsFutures);

        ResultType r = boost::move(HPX_STD_GET(1, compRes).move());
        res.second.insert(res.second.end(), r.second.begin(), r.second.end());
        res.first += r.first;
        componentsFutures.erase(componentsFutures.begin() + HPX_STD_GET(0, compRes));
    }

    return res;
}

}
}
}

#endif
