#ifndef LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <algorithm>

namespace LibGeoDecomp {

/**
 * This grid will rearrange cells in its delegate grid to match the
 * order defined by a compaction (defined by a node set) and the
 * primary weights matrix. The goal is to remove all unused elements
 * and make the order yielded from the partial sortation in the
 * SELL-C-SIGMA format match the physical memory layout.
 *
 * One size fits both, SoA and AoS -- although SIGMA > 1 is only
 * really relevant for SoA layouts.
 *
 * fixme: only implement this for SoA? AoS is slow anyway
 */
template<typename DELEGATE_GRID>
class ReorderingUnstructuredGrid : public GridBase<ELEMENT_TYPE, 1, WEIGHT_TYPE>
{
public:
    typedef DELEGATE_GRID::WeightType WeightType;
    const static int DIM = 1;
    const static int SIGMA = DELEGATE_GRID::SIGMA;

    explicit ReorderingUnstructuredGrid(
        const Region<1>& nodeSet) :
        nodeSet(nodeSet)
    {}

    // fixme: 1. finish GridBase interface with internal ID translation (via logicalToPhysicalID)
    // fixme: 2. ID translation is slow, but acceptable for IO. not acceptable for updates.
    //           probably acceptable for ghost zone communication.

    // fixme: 3. allow steppers, simulators to obtain a remapped
    // region from a grid (e.g. remapRegion(), needs to be in
    // gridBase) and add accessors that work with these remapped regions. can probably be inside specialized updatefunctor

    inline
    void setWeights(std::size_t matrixID, const std::map<Coord<2>, VALUE_TYPE>& matrix)
    {
        std::map<int, int> rowLenghts;
        for (std::map<Coord<2>, VALUE_TYPE>::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
            ++rowLenghts[i->first.x()];
        }

        typedef std::pair<int, int> IntPair;
        typedef std::vector<IntPair> RowLengthVec;
        RowLengthVec reorderedRowLengths;
        reorderedRowLengths.reserve(nodeSet.size());

        for (Region<1>::StreakIterator i = nodeSet.begin(); i != nodeSet.end(); ++i) {
            for (int j = i; j != i->endX; ++j) {
                reorderedRowLengths << std::make_pair(j, rowLenghts[j]);
            }
        }

        for (RowLengthVec::iterator i = reorderedRowLengths.begin(); i != reorderedRowLengths.end(); ) {
            RowLengthVec::iterator nextStop = std::min(i + SIGMA, reorderedRowLengths.end());

            std::stable_sort(i, nextStop, [](const IntPair& a, const IntPair& b) {
                    return a.second > b.second;
                });
        }

        logicalToPhysicalID.clear();
        physicalToLogicalID.clear();
        physicalToLogicalID.reserve(nodeSet.size());

        for (std::size_t i = 0; i < reorderedRowLengths.size(); ++i) {
            int logicalID = reorderedRowLengths[i].first;
            logicalToPhysicalID[logicalID] = i;
            physicalToLogicalID << logicalID;
        }
    }



    virtual void resize(const CoordBox<DIM>&)
    {
        // fixme: does this even make sense?
    }

    virtual void set(const Coord<DIM>&, const CELL&)
    {
        // fixme
    }

    virtual void set(const Streak<DIM>&, const CELL*)
    {
        // fixme
    }

    virtual CELL get(const Coord<DIM>&) const
    {
        // fixme
    }

    virtual void get(const Streak<DIM>&, CELL *) const
    {
        // fixme
    }

    virtual void setEdge(const CELL&)
    {
        // fixme
    }

    virtual const CELL& getEdge() const
    {
        // fixme
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        // fixme
    }

    virtual void saveRegion(std::vector<char> *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        // fixme: likely needs specialized re-implementation to carry
        // region and remapping facility into soa_grid::callback().
        // load/save need to observe ordering from original region to
        // avoid clashes with remote side.
    }

    virtual void loadRegion(const std::vector<char>& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        // fixme: likely needs specialized re-implementation to carry
        // region and remapping facility into soa_grid::callback().
        // load/save need to observe ordering from original region to
        // avoid clashes with remote side.
    }

private:
    Region<1> nodeSet;
    // fixme: this type is probably shite (huge memory overhead).
    // alternative: sorted std::vector<IntPair> also provides log(n)
    // lookup, but minimal memory overhead.
    std::map<int, int> logicalToPhysicalID;
    std::vector<int> physicalToLogicalID

    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        // fixme
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        // fixme
    }

};

}

#endif
#endif
