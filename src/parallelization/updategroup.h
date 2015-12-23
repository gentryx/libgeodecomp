#ifndef LIBGEODECOMP_PARALLELIZATION_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_UPDATEGROUP_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, template<typename GRID_TYPE> class PATCH_LINK>
class UpdateGroup
{
public:
    typedef LibGeoDecomp::HiParSimulator::Stepper<CELL_TYPE> StepperType;
    typedef typename StepperType::Topology Topology;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, true, SupportsSoA>::Value GridType;
    typedef typename PATCH_LINK<GridType>::Link PatchLink;
    typedef boost::shared_ptr<PatchLink> PatchLinkPtr;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;
    typedef typename StepperType::PatchAccepterVec PatchAccepterVec;
    typedef typename StepperType::PatchProviderVec PatchProviderVec;

    const static int DIM = Topology::DIM;

    UpdateGroup(
        unsigned ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        unsigned rank) :
        partitionManager(new PartitionManagerType()),
        ghostZoneWidth(ghostZoneWidth),
        initializer(initializer),
        rank(rank)
    {}

protected:
    std::vector<PatchLinkPtr> patchLinks;
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    unsigned rank;
};

}

#endif
