#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONMANAGER_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONMANAGER_H

#include <libgeodecomp/config.h>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/region.h>

namespace LibGeoDecomp {

namespace HiParSimulator {
class VanillaStepperTest;
}

template<typename TOPOLOGY>
class PartitionManager
{
public:
    friend class PartitionManagerTest;
    friend class HiParSimulator::VanillaStepperTest;
    typedef TOPOLOGY Topology;
    static const int DIM = Topology::DIM;
    typedef std::map<int, std::vector<Region<DIM> > > RegionVecMap;

    enum AccessCode {
        OUTGROUP = -1
    };

    PartitionManager(
        const CoordBox<DIM>& simulationArea=CoordBox<DIM>())
    {
        std::vector<std::size_t> weights(1, simulationArea.size());
        boost::shared_ptr<Partition<DIM> > partition(
            new StripingPartition<DIM>(Coord<DIM>(), simulationArea.dimensions, 0, weights));
        resetRegions(simulationArea, partition, 0, 1);
        resetGhostZones(std::vector<CoordBox<DIM> >(1));
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
            if (i != rank &&
                boundingBoxes[i].intersects(ownBoundingBox) &&
                (!(getRegion(rank, ghostZoneWidth) &
                   getRegion(i,    0)).empty() ||
                 !(getRegion(i,    ghostZoneWidth) &
                   getRegion(rank, 0)).empty())) {
                intersect(i);
            }
        }

        // outgroup ghost zone fragments are computed a tad generous,
        // an exact, greedy calculation would be more complicated
        Region<DIM> outer = outerRim;
        Region<DIM> inner = rim(getGhostZoneWidth());
        for (typename RegionVecMap::iterator i = outerGhostZoneFragments.begin();
             i != outerGhostZoneFragments.end();
             ++i) {
            if (i->first != OUTGROUP) {
                outer -= i->second.back();
            }
        }
        for (typename RegionVecMap::iterator i = innerGhostZoneFragments.begin();
             i != innerGhostZoneFragments.end();
             ++i) {
            if (i->first != OUTGROUP) {
                inner -= i->second.back();
            }
        }
        outerGhostZoneFragments[OUTGROUP] =
            std::vector<Region<DIM> >(getGhostZoneWidth() + 1, outer);
        innerGhostZoneFragments[OUTGROUP] =
            std::vector<Region<DIM> >(getGhostZoneWidth() + 1, inner);
    }

    inline RegionVecMap& getOuterGhostZoneFragments()
    {
        return outerGhostZoneFragments;
    }

    inline RegionVecMap& getInnerGhostZoneFragments()
    {
        return innerGhostZoneFragments;
    }

    inline const Region<DIM>& getInnerOutgroupGhostZoneFragment()
    {
        return innerGhostZoneFragments[OUTGROUP].back();
    }

    inline const Region<DIM>& getOuterOutgroupGhostZoneFragment()
    {
        return outerGhostZoneFragments[OUTGROUP].back();
    }

    inline const Region<DIM>& getRegion(
        int node,
        unsigned expansionWidth)
    {
        if (regions.count(node) == 0) {
            fillRegion(node);
        }
        return regions[node][expansionWidth];
    }

    inline const Region<DIM>& ownRegion(unsigned expansionWidth = 0)
    {
        return regions[rank][expansionWidth];
    }

    inline const Region<DIM>& ownExpandedRegion()
    {
        return regions[rank].back();
    }

    inline const Region<DIM>& rim(unsigned dist)
    {
        return ownRims[dist];
    }

    inline const Region<DIM>& innerSet(unsigned dist)
    {
        return ownInnerSets[dist];
    }

    inline const std::vector<CoordBox<DIM> >& getBoundingBoxes() const
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

    inline const Region<DIM>& getInnerRim() const
    {
        return innerRim;
    }

    inline const std::vector<std::size_t>& getWeights() const
    {
        return partition->getWeights();
    }

private:
    boost::shared_ptr<Partition<DIM> > partition;
    CoordBox<DIM> simulationArea;
    Region<DIM> outerRim;
    Region<DIM> volatileKernel;
    Region<DIM> innerRim;
    RegionVecMap regions;
    RegionVecMap outerGhostZoneFragments;
    RegionVecMap innerGhostZoneFragments;
    std::vector<Region<DIM> > ownRims;
    std::vector<Region<DIM> > ownInnerSets;
    unsigned rank;
    unsigned ghostZoneWidth;
    std::vector<CoordBox<DIM> > boundingBoxes;

    inline void fillRegion(unsigned node)
    {
        std::vector<Region<DIM> >& regionExpansion = regions[node];
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
        for (int i = getGhostZoneWidth() - 1; i >= 0; --i) {
            ownRims[i] = ownRims[i + 1].expandWithTopology(
                1, simulationArea.dimensions, Topology());
        }

        ownInnerSets.front() = ownRegion();
        Region<DIM> minuend = surface.expandWithTopology(
            1, simulationArea.dimensions, Topology());
        for (std::size_t i = 1; i <= getGhostZoneWidth(); ++i) {
            ownInnerSets[i] = ownInnerSets[i - 1] - minuend;
            minuend = minuend.expandWithTopology(
                1, simulationArea.dimensions, Topology());
        }

        volatileKernel = ownInnerSets.back() & rim(1) ;
        innerRim       = ownInnerSets.back() & rim(0) ;
    }

    inline void intersect(unsigned node)
    {
        std::vector<Region<DIM> >& outerGhosts = outerGhostZoneFragments[node];
        std::vector<Region<DIM> >& innerGhosts = innerGhostZoneFragments[node];
        outerGhosts.resize(getGhostZoneWidth() + 1);
        innerGhosts.resize(getGhostZoneWidth() + 1);
        for (unsigned i = 0; i <= getGhostZoneWidth(); ++i) {
            outerGhosts[i] = getRegion(rank, i) & getRegion(node, 0);
            innerGhosts[i] = getRegion(rank, 0) & getRegion(node, i);
        }
    }
};

}

#endif
