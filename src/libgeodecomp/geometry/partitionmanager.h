#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONMANAGER_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONMANAGER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/dummyadjacencymanufacturer.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/misc/sharedptr.h>

namespace LibGeoDecomp {

/**
 * The PartitionManager maintains the Regions which describe a node's
 * subdomain (as defined by a Partition) and the inner and outer ghost
 * regions (halos) which are used for synchronization with neighboring
 * subdomains.
 */
template<typename TOPOLOGY>
class PartitionManager
{
public:
    friend class PartitionManagerTest;
    friend class VanillaStepperTest;

    typedef TOPOLOGY Topology;
    static const int DIM = Topology::DIM;
    typedef std::map<int, std::vector<Region<DIM> > > RegionVecMap;

    enum AccessCode {
        OUTGROUP = -1
    };

    explicit PartitionManager(
        const CoordBox<DIM>& simulationArea = CoordBox<DIM>())
    {
        std::vector<std::size_t> weights(1, simulationArea.size());
        typename SharedPtr<Partition<DIM> >::Type partition(
            new StripingPartition<DIM>(
                Coord<DIM>(),
                simulationArea.dimensions,
                0,
                weights));

        resetRegions(
            makeShared(new DummyAdjacencyManufacturer<DIM>()),
            simulationArea,
            partition,
            0,
            1);

        resetGhostZones(std::vector<CoordBox<DIM> >(1), std::vector<CoordBox<DIM> >(1));
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
        typename SharedPtr<AdjacencyManufacturer<DIM> >::Type newAdjacencyManufacturer,
        const CoordBox<DIM>& newSimulationArea,
        typename SharedPtr<Partition<DIM> >::Type newPartition,
        unsigned newRank,
        unsigned newGhostZoneWidth)
    {
        adjacencyManufacturer = newAdjacencyManufacturer;
        partition = newPartition;
        simulationArea = newSimulationArea;
        myRank = newRank;
        ghostZoneWidth = newGhostZoneWidth;
        regions.clear();
        outerGhostZoneFragments.clear();
        innerGhostZoneFragments.clear();
        fillOwnRegion();
    }

    inline void resetGhostZones(
        const std::vector<CoordBox<DIM> >& newBoundingBoxes,
        const std::vector<CoordBox<DIM> >& newExpandedBoundingBoxes)
    {
        if (newBoundingBoxes.size() != newExpandedBoundingBoxes.size()) {
            throw std::logic_error("number of bounding boxes doesn't match");
        }

        boundingBoxes = newBoundingBoxes;
        expandedBoundingBoxes = newExpandedBoundingBoxes;
        CoordBox<DIM> ownBoundingBox = ownRegion().boundingBox();
        CoordBox<DIM> ownExpandedBoundingBox = ownExpandedRegion().boundingBox();

        for (int i = 0; i < static_cast<std::ptrdiff_t>(boundingBoxes.size()); ++i) {
            if ((i != myRank) &&
                (boundingBoxes[i].intersects(ownExpandedBoundingBox) ||
                 expandedBoundingBoxes[i].intersects(ownBoundingBox)) &&
                (!(getRegion(myRank, ghostZoneWidth) &
                   getRegion(i,      0)).empty() ||
                 !(getRegion(i,      ghostZoneWidth) &
                   getRegion(myRank, 0)).empty())) {
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
        return regions[myRank][expansionWidth];
    }

    inline const Region<DIM>& ownExpandedRegion()
    {
        return regions[myRank].back();
    }

    /**
     * Rim describes the node's inner ghost zone and those surrounding
     * coordinates required to update those.
     */
    inline const Region<DIM>& rim(unsigned dist)
    {
        return ownRims[dist];
    }

    /**
     * inner set refers to that part of a node's domain which are
     * required to update the kernel.
     */
    inline const Region<DIM>& innerSet(unsigned dist)
    {
        return ownInnerSets[dist];
    }

    inline unsigned getGhostZoneWidth() const
    {
        return ghostZoneWidth;
    }

    /**
     * outer rim is the union of all outer ghost zone fragments.
     */
    inline const Region<DIM>& getOuterRim() const
    {
        return outerRim;
    }

    /**
     * The volatile kernel is the part of the kernel which may be
     * overwritten while updating the inner ghost zone.
     */
    inline const Region<DIM>& getVolatileKernel() const
    {
        return volatileKernel;
    }

    /**
     * The inner rim is the part of the kernel which is required for
     * updating the own rims. Its similar to the outer ghost zone, but
     * to the inner side. It usually includes just one stencil
     * diameter more cells than the volatile kernel.
     */
    inline const Region<DIM>& getInnerRim() const
    {
        return innerRim;
    }

    inline const std::vector<std::size_t>& getWeights() const
    {
        return partition->getWeights();
    }

    inline int rank() const
    {
        return myRank;
    }

    inline const Coord<DIM>& getSimulationArea() const
    {
        return simulationArea.dimensions;
    }

private:
    typename SharedPtr<AdjacencyManufacturer<DIM> >::Type adjacencyManufacturer;
    typename SharedPtr<Partition<DIM> >::Type partition;
    CoordBox<DIM> simulationArea;
    Region<DIM> outerRim;
    Region<DIM> volatileKernel;
    Region<DIM> innerRim;
    RegionVecMap regions;
    RegionVecMap outerGhostZoneFragments;
    RegionVecMap innerGhostZoneFragments;
    std::vector<Region<DIM> > ownRims;
    std::vector<Region<DIM> > ownInnerSets;
    int myRank;
    unsigned ghostZoneWidth;
    std::vector<CoordBox<DIM> > boundingBoxes;
    std::vector<CoordBox<DIM> > expandedBoundingBoxes;

    const SharedPtr<Adjacency>::Type adjacency(const Region<DIM>& region) const
    {
        return adjacencyManufacturer->getAdjacency(region);
    }

    const SharedPtr<Adjacency>::Type reverseAdjacency(const Region<DIM>& region) const
    {
        return adjacencyManufacturer->getReverseAdjacency(region);
    }

    inline void fillRegion(int node)
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
                Topology(),
                *adjacency(reg));
            regionExpansion[i] = expanded;
        }
    }

    inline void fillOwnRegion()
    {
        fillRegion(myRank);
        Region<DIM> surface(
            ownRegion().expandWithTopology(
                1,
                simulationArea.dimensions,
                Topology(),
                *reverseAdjacency(ownRegion())) - ownRegion());
        Region<DIM> kernel(
            ownRegion() -
            surface.expandWithTopology(
                getGhostZoneWidth(),
                simulationArea.dimensions,
                Topology(),
                *adjacency(surface)));
        outerRim = ownExpandedRegion() - ownRegion();
        ownRims.resize(getGhostZoneWidth() + 1);
        ownInnerSets.resize(getGhostZoneWidth() + 1);

        ownRims.back() = ownRegion() - kernel;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(getGhostZoneWidth() - 1); i >= 0; --i) {
            std::size_t index = static_cast<std::size_t>(i);
            ownRims[index] = ownRims[index + 1].expandWithTopology(
                1,
                simulationArea.dimensions,
                Topology(),
                *adjacency(ownRims[index + 1]));
        }

        ownInnerSets[getGhostZoneWidth()] = kernel;
        for (std::size_t i = getGhostZoneWidth(); i > 0; --i) {
            ownInnerSets[i - 1] = ownInnerSets[i].expandWithTopology(
                1,
                simulationArea.dimensions,
                Topology(),
                *adjacency(ownInnerSets[i]));
        }

        volatileKernel = ownInnerSets.back() & rim(0);
        innerRim       = ownInnerSets.back() & rim(0);
    }

    inline void intersect(int node)
    {
        std::vector<Region<DIM> >& outerGhosts = outerGhostZoneFragments[node];
        std::vector<Region<DIM> >& innerGhosts = innerGhostZoneFragments[node];
        outerGhosts.resize(getGhostZoneWidth() + 1);
        innerGhosts.resize(getGhostZoneWidth() + 1);

        bool outerFragmentsAllEmpty = true;
        bool innerFragmentsAllEmpty = true;

        for (unsigned i = 0; i <= getGhostZoneWidth(); ++i) {
            outerGhosts[i] = getRegion(myRank, i) & getRegion(node, 0);
            innerGhosts[i] = getRegion(myRank, 0) & getRegion(node, i);

            outerFragmentsAllEmpty &= outerGhosts[i].empty();
            innerFragmentsAllEmpty &= innerGhosts[i].empty();
        }

        if (outerFragmentsAllEmpty) {
            outerGhostZoneFragments.erase(node);
        }

        if (innerFragmentsAllEmpty) {
            innerGhostZoneFragments.erase(node);
        }
    }
};

}

#endif
