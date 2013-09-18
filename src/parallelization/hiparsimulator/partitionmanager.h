#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONMANAGER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONMANAGER_H

#include <libgeodecomp/config.h>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: get rid of DIM, deduce it from TOPOLOGY
template<int DIM, typename TOPOLOGY=typename Topologies::Cube<DIM>::Topology>
class PartitionManager {
    friend class HiParSimulatorTest;
    friend class PartitionManagerTest;
    friend class VanillaStepperTest;
public:
    typedef SuperMap<int, SuperVector<Region<DIM> > > RegionVecMap;
    typedef TOPOLOGY Topology;

    enum AccessCode {OUTGROUP = -1};

    PartitionManager(
        const CoordBox<DIM>& simulationArea=CoordBox<DIM>())
    {
        SuperVector<std::size_t> weights(1, simulationArea.size());
        boost::shared_ptr<Partition<DIM> > partition(
            new StripingPartition<DIM>(Coord<DIM>(), simulationArea.dimensions, 0, weights));
        resetRegions(simulationArea, partition, 0, 1);
        resetGhostZones(SuperVector<CoordBox<DIM> >(1));
    }

    /**
     * resets the domain decomposition. The simulation space is
     * described by newSimulationArea, the decomposition scheme by
     * newPartition. newRank will usually correspond to the MPI rank
     * and identifies the current process. newGhostZoneWidth specifies
     * after how many steps the halo should be synchronized. Higher
     * values mean that the halo will be wider, which requires fewer
     * synchronizations, but the syncs need to communicate more data.
     * This is primarily to combat high latency datapaths (e.g.
     * network latency or if the data needs to go to remote
     * accelerators).
     */
    inline void resetRegions(
        const CoordBox<DIM>& newSimulationArea,
        boost::shared_ptr<Partition<DIM> > newPartition,
        unsigned newRank,
        unsigned newGhostZoneWidth)
    {
        partition = newPartition;
        simulationArea = newSimulationArea;
        rank = newRank;
        ghostZoneWidth = newGhostZoneWidth;
        regions.clear();
        outerGhostZoneFragments.clear();
        innerGhostZoneFragments.clear();
        fillOwnRegion();
    }

    inline void resetGhostZones(
        const std::vector<CoordBox<DIM> >& newBoundingBoxes)
    {
        boundingBoxes = newBoundingBoxes;
        CoordBox<DIM> ownBoundingBox = ownExpandedRegion().boundingBox();

        for (unsigned i = 0; i < boundingBoxes.size(); ++i) {
            if (i != rank)
                intersect(i);
        }

        // outgroup ghost zone fragments are computed a tad generous,
        // an exact, greedy calculation would be more complicated
        Region<DIM> outer = outerRim;
        Region<DIM> inner = rim(getGhostZoneWidth());
        for (typename RegionVecMap::iterator i = outerGhostZoneFragments.begin();
             i != outerGhostZoneFragments.end();
             ++i)
            if (i->first != OUTGROUP && !i->second.empty())
                outer -= i->second.back();
        for (typename RegionVecMap::iterator i = innerGhostZoneFragments.begin();
             i != innerGhostZoneFragments.end();
             ++i)
            if (i->first != OUTGROUP && !i->second.empty())
                inner -= i->second.back();
        outerGhostZoneFragments[OUTGROUP] =
            SuperVector<Region<DIM> >(getGhostZoneWidth() + 1, outer);
        innerGhostZoneFragments[OUTGROUP] =
            SuperVector<Region<DIM> >(getGhostZoneWidth() + 1, inner);
    }

    inline const RegionVecMap& getOuterGhostZoneFragments() const
    {
        return outerGhostZoneFragments;
    }

    inline const RegionVecMap& getInnerGhostZoneFragments() const
    {
        return innerGhostZoneFragments;
    }

    inline const Region<DIM>& getInnerOutgroupGhostZoneFragment() const
    {
        return innerGhostZoneFragments[OUTGROUP].back();
    }

    inline const Region<DIM>& getOuterOutgroupGhostZoneFragment() const
    {
        return outerGhostZoneFragments[OUTGROUP].back();
    }

    inline const Region<DIM>& getRegion(
        int node,
        unsigned expansionWidth)
    {
        if (regions.count(node) == 0)
            fillRegion(node);
        return regions[node][expansionWidth];
    }

    inline const Region<DIM>& ownRegion(unsigned expansionWidth = 0) const
    {
        return regions[rank][expansionWidth];
    }

    inline const Region<DIM>& ownExpandedRegion() const
    {
        return regions[rank].back();
    }

    inline const Region<DIM>& rim(unsigned dist) const
    {
        return ownRims[dist];
    }

    inline const Region<DIM>& innerSet(unsigned dist) const
    {
        return ownInnerSets[dist];
    }

    inline const SuperVector<CoordBox<DIM> >& getBoundingBoxes() const
    {
        return boundingBoxes;
    }

    inline unsigned getGhostZoneWidth() const
    {
        return ghostZoneWidth;
    }

    inline const Region<DIM>& getOuterRim() const
    {
        return outerRim;
    }

    inline const Region<DIM>& getVolatileKernel() const
    {
        return volatileKernel;
    }

    inline const SuperVector<std::size_t>& getWeights() const
    {
        return partition->getWeights();
    }

private:
    boost::shared_ptr<Partition<DIM> > partition;
    CoordBox<DIM> simulationArea;
    Region<DIM> outerRim;
    Region<DIM> volatileKernel;
    RegionVecMap regions;
    RegionVecMap outerGhostZoneFragments;
    RegionVecMap innerGhostZoneFragments;
    SuperVector<Region<DIM> > ownRims;
    SuperVector<Region<DIM> > ownInnerSets;
    unsigned rank;
    unsigned ghostZoneWidth;
    SuperVector<CoordBox<DIM> > boundingBoxes;

    inline void fillRegion(unsigned node)
    {
        SuperVector<Region<DIM> >& regionExpansion = regions[node];
        regionExpansion.resize(getGhostZoneWidth() + 1);
        regionExpansion[0] = partition->getRegion(node);
        for (std::size_t i = 1; i <= getGhostZoneWidth(); ++i) {
            Region<DIM> expanded;
            const Region<DIM>& reg = regionExpansion[i - 1];
            expanded = reg.expandWithTopology(
                1,
                simulationArea.dimensions,
                Topology());
            regionExpansion[i] = expanded;
        }
    }

    inline void fillOwnRegion()
    {
        fillRegion(rank);
        Region<DIM> surface(
            ownRegion().expandWithTopology(
                1, simulationArea.dimensions, Topology()) -
            ownRegion());
        Region<DIM> kernel(
            ownRegion() -
            surface.expandWithTopology(
                getGhostZoneWidth(),
                simulationArea.dimensions,
                Topology()));
        outerRim = ownExpandedRegion() - ownRegion();
        ownRims.resize(getGhostZoneWidth() + 1);
        ownInnerSets.resize(getGhostZoneWidth() + 1);

        ownRims.back() = ownRegion() - kernel;
        for (int i = getGhostZoneWidth() - 1; i >= 0; --i)
            ownRims[i] = ownRims[i + 1].expandWithTopology(
                1, simulationArea.dimensions, Topology());

        ownInnerSets.front() = ownRegion();
        Region<DIM> minuend = surface.expandWithTopology(
            1, simulationArea.dimensions, Topology());
        for (std::size_t i = 1; i <= getGhostZoneWidth(); ++i) {
            ownInnerSets[i] = ownInnerSets[i - 1] - minuend;
            minuend = minuend.expandWithTopology(
                1, simulationArea.dimensions, Topology());
        }

        volatileKernel = ownInnerSets.back() & rim(1) ;
    }

    inline void intersect(unsigned node)
    {
        SuperVector<Region<DIM> >& outerGhosts = outerGhostZoneFragments[node];
        SuperVector<Region<DIM> >& innerGhosts = innerGhostZoneFragments[node];
        outerGhosts.reserve(getGhostZoneWidth() + 1);
        innerGhosts.reserve(getGhostZoneWidth() + 1);
        for (unsigned i = 0; i <= getGhostZoneWidth(); ++i) {
            const Region<DIM>& outerGhost = getRegion(rank, i) & getRegion(node, 0);
            if(!outerGhost.empty())
                outerGhosts << outerGhost;
            const Region<DIM>& innerGhost = getRegion(rank, 0) & getRegion(node, i);
            if(!innerGhost.empty())
                innerGhosts << innerGhost;
        }
    }
};

}
}

#endif
