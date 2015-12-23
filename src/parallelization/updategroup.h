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

    typedef typename StepperType::PatchType PatchType;
    typedef typename StepperType::PatchProviderPtr PatchProviderPtr;
    typedef typename StepperType::PatchAccepterPtr PatchAccepterPtr;

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

    virtual ~UpdateGroup()
    {
        for (typename std::vector<PatchLinkPtr>::iterator i = patchLinks.begin();
             i != patchLinks.end();
             ++i) {
            (*i)->cleanup();
        }
    }

    const Chronometer& statistics() const
    {
        return stepper->statistics();
    }

    void addPatchProvider(
        const PatchProviderPtr& patchProvider,
        const PatchType& patchType)
    {
        stepper->addPatchProvider(patchProvider, patchType);
    }

    void addPatchAccepter(
        const PatchAccepterPtr& patchAccepter,
        const PatchType& patchType)
    {
        stepper->addPatchAccepter(patchAccepter, patchType);
    }

    inline void update(int nanoSteps)
    {
        stepper->update(nanoSteps);
    }

    const GridType& grid() const
    {
        return stepper->grid();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return stepper->currentStep();
    }

    inline const std::vector<std::size_t>& getWeights() const
    {
        return partitionManager->getWeights();
    }

    inline double computeTimeInner() const
    {
        return stepper->computeTimeInner;
    }

    inline double computeTimeGhost() const
    {
        return stepper->computeTimeGhost;
    }

    inline double patchAcceptersTime() const
    {
        return stepper->patchAcceptersTime;
    }

    inline double patchProvidersTime() const
    {
        return stepper->patchAcceptersTime;
    }

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
