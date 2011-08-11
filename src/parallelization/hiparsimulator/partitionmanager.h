#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_partitionmanager_h_
#define _libgeodecomp_parallelization_hiparsimulator_partitionmanager_h_
// fixme: fix _hiparsimulator prefix in header guards, in this file and others

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename TOPOLOGY>
class TopologyWrapsXAxis
{
public:
    bool operator()()
    {
        if (TOPOLOGY::DIMENSIONS == 1) {
            return TOPOLOGY::WrapEdges;
        } else {
            return TopologyWrapsXAxis<typename TOPOLOGY::ParentTopology>()();
        }
    }
};

template<>
class TopologyWrapsXAxis<Topologies::ZeroDimensional>
{
public:
    bool operator()()
    {
        return 0;
    }
};

template<int DIM, typename TOPOLOGY=typename Topologies::Cube<DIM>::Topology>
class PartitionManager {
    friend class PartitionManagerTest;
    friend class VanillaStepperTest;
    template<int> friend class RimMarker;
    template<int> friend class InnerSetMarker;
    friend class HiParSimulatorTest;
public:
    typedef SuperMap<int, SuperVector<Region<DIM> > > RegionVecMap;

    enum AccessCode {OUTGROUP = -1};

    PartitionManager(
        const CoordBox<DIM>& _simulationArea=CoordBox<DIM>())
    {
        SuperVector<unsigned> weights(1, _simulationArea.size());
        StripingPartition<DIM> partition(
            Coord<DIM>(), _simulationArea.dimensions);
        VanillaRegionAccumulator<StripingPartition<DIM>, DIM> *accu = 
            new VanillaRegionAccumulator<StripingPartition<DIM>, DIM>(
                partition,
                0,
                weights);
        resetRegions(_simulationArea, accu, 0, 1);
        resetGhostZones(SuperVector<CoordBox<DIM> >(1));
    }

    inline void resetRegions(
        const CoordBox<DIM>& _simulationArea,
        RegionAccumulator<DIM> *_regionAccu,
        const unsigned& _rank,
        const unsigned& _ghostZoneWidth) 
    {
        regionAccu.reset(_regionAccu);
        simulationArea = Region<DIM>();
        simulationArea << _simulationArea;
        rank = _rank;
        ghostZoneWidth = _ghostZoneWidth;
        regions.clear();
        outerGhostZoneFragments.clear();
        innerGhostZoneFragments.clear();
        fillOwnRegion();
    }

    inline void resetGhostZones(
        const SuperVector<CoordBox<DIM> >& _boundingBoxes)
    {
        boundingBoxes = _boundingBoxes;
        CoordBox<DIM> ownBoundingBox = ownExpandedRegion().boundingBox();

        for (unsigned i = 0; i < boundingBoxes.size(); ++i) 
            if (i != rank && 
                boundingBoxes[i].intersects(ownBoundingBox) &&
                (!(getRegion(rank, ghostZoneWidth) & getRegion(i,    0)).empty() ||
                 !(getRegion(i,    ghostZoneWidth) & getRegion(rank, 0)).empty())) 
                intersect(i); 

        // outgroup ghost zone fragments are computed a tad generous,
        // an exact, greedy calculation would be more complicated
        Region<DIM> outer = outerRim;
        Region<DIM> inner = innerRim;
        for (typename RegionVecMap::iterator i = outerGhostZoneFragments.begin();
             i != outerGhostZoneFragments.end();
             ++i)
            if (i->first != OUTGROUP)
                outer -= i->second.back();
        for (typename RegionVecMap::iterator i = innerGhostZoneFragments.begin();
             i != innerGhostZoneFragments.end();
             ++i)
            if (i->first != OUTGROUP)
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
        const int& node, 
        const unsigned& expansionWidth)
    {
        if (regions.count(node) == 0) 
            fillRegion(node);
        return regions[node][expansionWidth];
    }
    
    inline const Region<DIM>& ownRegion(const unsigned& expansionWidth = 0) 
    {
        return regions[rank][expansionWidth];
    }

    inline const Region<DIM>& ownExpandedRegion()
    {
        return regions[rank].back();
    }

    inline const Region<DIM>& rim(const unsigned& dist) const
    {
        return ownRims[dist];
    }

    inline const Region<DIM>& innerSet(const unsigned& dist) const
    {
        return ownInnerSets[dist];
    }
    
    inline const SuperVector<CoordBox<DIM> >& getBoundingBoxes() const
    {
        return boundingBoxes;
    }

    inline const unsigned& getGhostZoneWidth() const
    {
        return ghostZoneWidth;
    }

    inline const Region<DIM>& getOuterRim() const
    {
        return outerRim;
    }
    
private:
    inline void fillRegion(const unsigned& node)
    {
        SuperVector<Region<DIM> >& regionExpansion = regions[node];
        regionExpansion.resize(getGhostZoneWidth() + 1);
        regionExpansion[0] = regionAccu->getRegion(node);
        for (int i = 1; i <= getGhostZoneWidth(); ++i) {
            Region<DIM> expanded = regionExpansion[i - 1].expand(1);
            Region<DIM> trimmed;

            for (StreakIterator<DIM> s = expanded.beginStreak(); 
                 s != expanded.endStreak(); ++s) {
                
                if (TopologyWrapsXAxis<TOPOLOGY>()()) {
                    splitStreak(*s, &trimmed);
                } else {
                    trimmed << *s;
                }
            }

            regionExpansion[i] = trimmed & simulationArea;
        }
    }

    void splitStreak(const Streak<DIM>& streak, Region<DIM> *target)
    {
        int width = simulationArea.boundingBox().dimensions.x();

        int currentX = streak.origin.x();
        if (currentX < 0) {
            Streak<DIM> section = streak;
            section.endX = std::min(streak.endX, 0);
            currentX = section.endX;

            // normalize left overhang
            section.origin.x() += width;
            section.endX += width;
            normalizeStreak(section, target);
        }

        if (currentX < streak.endX) {
            Streak<DIM> section = streak;
            section.origin.x() = currentX;
            section.endX = std::min(streak.endX, width);
            currentX = section.endX;

            normalizeStreak(section, target);
        }

        if (currentX < streak.endX) {
            Streak<DIM> section = streak;
            section.origin.x() = currentX;

            // normalize right overhang
            section.origin.x() -= width;
            section.endX -= width;
            normalizeStreak(section, target);
        }
    }
    
    void normalizeStreak(const Streak<DIM>& streak, Region<DIM> *target)
    {
        Coord<DIM> c;
        Streak<DIM> ret;
        CoordNormalizer<DIM, DIM> normalizer(
            &c, 
            simulationArea.boundingBox().dimensions);
        ret.origin = TOPOLOGY::locate(normalizer, streak.origin);
        ret.endX = ret.origin.x() + streak.length();

        if (ret.origin != normalizer.getEdgeCell())
            (*target) << ret;
    }

    inline void fillOwnRegion()
    {
        fillRegion(rank);
        Region<DIM> rim((ownRegion().expand(1) - ownRegion()) & simulationArea);
        Region<DIM> kernel(ownRegion() - rim.expand(getGhostZoneWidth()));
        innerRim = ownRegion() - kernel;
        outerRim = ownExpandedRegion() - ownRegion();
        ownRims.resize(getGhostZoneWidth() + 1);
        ownInnerSets.resize(getGhostZoneWidth() + 1);

        ownRims.back() = innerRim;
        for (int i = getGhostZoneWidth() - 1; i >= 0; --i)
            ownRims[i] = ownRims[i + 1].expand() & simulationArea;

        ownInnerSets.front() = ownRegion();
        Region<DIM> minuend = rim.expand();
        for (int i = 1; i <= getGhostZoneWidth(); ++i) {
            ownInnerSets[i] = ownInnerSets[i - 1] - minuend;
            minuend = minuend.expand();
        }
    }

    inline void intersect(const unsigned& node) 
    {
        SuperVector<Region<DIM> >& outerGhosts = outerGhostZoneFragments[node];
        SuperVector<Region<DIM> >& innerGhosts = innerGhostZoneFragments[node];
        outerGhosts.resize(getGhostZoneWidth() + 1);
        innerGhosts.resize(getGhostZoneWidth() + 1);
        for (int i = 0; i <= getGhostZoneWidth(); ++i) {
            outerGhosts[i] = getRegion(rank, i) & getRegion(node, 0);
            innerGhosts[i] = getRegion(rank, 0) & getRegion(node, i);
        }
    }

private:
    boost::shared_ptr<RegionAccumulator<DIM> > regionAccu;
    Region<DIM> simulationArea;
    //fixme: remove inner/outer rim? isn't innerrim covered in ownRims anyways?
    Region<DIM> outerRim;
    Region<DIM> innerRim;
    Region<DIM> groupRegion;
    RegionVecMap regions;
    RegionVecMap outerGhostZoneFragments;
    RegionVecMap innerGhostZoneFragments;
    SuperVector<Region<DIM> > ownRims;
    SuperVector<Region<DIM> > ownInnerSets;
    unsigned rank;
    unsigned ghostZoneWidth;
    SuperVector<CoordBox<DIM> > boundingBoxes;
};

}
}

#endif
#endif
