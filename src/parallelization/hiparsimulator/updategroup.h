#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_updategroup_h_
#define _libgeodecomp_parallelization_hiparsimulator_updategroup_h_

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/intersectingregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchlink.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: STEPPER does not have to be a template parameter. it can also be defined solely in the constructor
template<class CELL_TYPE, class PARTITION, class STEPPER=VanillaStepper<CELL_TYPE> >
class UpdateGroup
{
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;
    typedef DisplacedGrid<
        CELL_TYPE, typename CELL_TYPE::Topology, true> GridType;
    typedef typename Stepper<CELL_TYPE>::PatchType PatchType;
    typedef typename Stepper<CELL_TYPE>::PatchProviderPtr PatchProviderPtr;
    typedef typename Stepper<CELL_TYPE>::PatchAccepterPtr PatchAccepterPtr;
    typedef boost::shared_ptr<typename PatchLink<GridType>::Link> PatchLinkPtr;
    typedef PartitionManager<DIM, typename CELL_TYPE::Topology> MyPartitionManager;
    typedef typename MyPartitionManager::RegionVecMap RegionVecMap;
    typedef typename Stepper<CELL_TYPE>::PatchAccepterVec PatchAccepterVec;

    UpdateGroup(
        const PARTITION& _partition, 
        const SuperVector<long>& _weights, 
        const unsigned& _offset,
        const CoordBox<DIM>& box, 
        const unsigned& _ghostZoneWidth,
        Initializer<CELL_TYPE> *_initializer,
        PatchAccepterVec patchAcceptersGhost=PatchAccepterVec(),
        PatchAccepterVec patchAcceptersInner=PatchAccepterVec(),
        const MPI::Datatype& _cellMPIDatatype = Typemaps::lookup<CELL_TYPE>(),
        MPI::Comm *communicator = &MPI::COMM_WORLD) : 
        partition(_partition),
        weights(_weights),
        offset(_offset),
        ghostZoneWidth(_ghostZoneWidth),
        initializer(_initializer),
        mpiLayer(communicator),
        cellMPIDatatype(_cellMPIDatatype),
        rank(mpiLayer.rank())
    {
        partitionManager.reset(new MyPartitionManager());
        partitionManager->resetRegions(
            box,
            new VanillaRegionAccumulator<PARTITION>(
                partition,
                offset,
                weights),
            rank,
            ghostZoneWidth);
        SuperVector<CoordBox<DIM> > boundingBoxes(mpiLayer.size());
        CoordBox<DIM> ownBoundingBox(partitionManager->ownRegion().boundingBox());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager->resetGhostZones(boundingBoxes);
        long firstSyncPoint =  
            initializer->startStep() * CELL_TYPE::nanoSteps() + ghostZoneWidth;

        // we have to hand over a list of all ghostzone senders as the
        // stepper will perform an initial update of the ghostzones
        // upon creation and we have to send those over to our neighbors.
        PatchAccepterVec ghostZoneAccepterLinks;
        RegionVecMap map = partitionManager->getInnerGhostZoneFragments();
        for (typename RegionVecMap::iterator i = map.begin(); i != map.end(); ++i) {
            if (!i->second.back().empty()) {
                boost::shared_ptr<typename PatchLink<GridType>::Accepter> link(
                    new typename PatchLink<GridType>::Accepter(
                        i->second.back(), 
                        i->first, 
                        0,
                        cellMPIDatatype, 
                        mpiLayer.getCommunicator()));
                ghostZoneAccepterLinks << link;
                patchLinks << link;

                link->charge(
                    firstSyncPoint, 
                    PatchLink<GridType>::ENDLESS, 
                    ghostZoneWidth);
            }
        }

        stepper.reset(new STEPPER(
                          partitionManager, 
                          initializer,
                          patchAcceptersGhost + ghostZoneAccepterLinks,
                          patchAcceptersInner));

        // the ghostzone receivers may be safely added after
        // initialization as they're only really needed when the next
        // ghostzone generation is being received.
        map = partitionManager->getOuterGhostZoneFragments();
        for (typename RegionVecMap::iterator i = map.begin(); i != map.end(); ++i) {
            if (!i->second.back().empty()) {
                boost::shared_ptr<typename PatchLink<GridType>::Provider> link(
                    new typename PatchLink<GridType>::Provider(
                        i->second.back(), 
                        i->first,
                        0,
                        cellMPIDatatype, 
                        mpiLayer.getCommunicator()));
                addPatchProvider(link, Stepper<CELL_TYPE>::GHOST);
                patchLinks << link;
         
                link->charge(
                    firstSyncPoint, 
                    PatchLink<GridType>::ENDLESS, 
                    ghostZoneWidth);
            }
        }
    }

    virtual ~UpdateGroup()
    { }

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

private:
    boost::shared_ptr<Stepper<CELL_TYPE> > stepper;
    boost::shared_ptr<MyPartitionManager> partitionManager;
    SuperVector<PatchLinkPtr> patchLinks;
    PARTITION partition;
    SuperVector<long> weights;
    unsigned offset;
    unsigned ghostZoneWidth;
    Initializer<CELL_TYPE> *initializer;
    MPILayer mpiLayer;
    MPI::Datatype cellMPIDatatype;
    unsigned rank;
};

}
}

#endif
#endif
