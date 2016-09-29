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
 * This reordering slows down get()/set() and possibly
 * loadRegion()/saveRegion(). Cell updates should not be harmed as the
 * update functor needs to use reordered Regions anyway.
 *
 * One size fits both, SoA and AoS. SIGMA > 1 is only really relevant
 * for SoA layouts, but compaction benefits both.
 */
template<typename DELEGATE_GRID>
class ReorderingUnstructuredGrid : public GridBase<typename DELEGATE_GRID::CellType, 1, typename DELEGATE_GRID::WeightType>
{
public:
    typedef typename DELEGATE_GRID::CellType CellType;
    typedef typename DELEGATE_GRID::WeightType WeightType;
    typedef std::pair<int, int> IntPair;

    const static int DIM = 1;
    const static int SIGMA = DELEGATE_GRID::SIGMA;

    explicit ReorderingUnstructuredGrid(
        const Region<1>& nodeSet) :
        nodeSet(nodeSet)
    {
        int physicalID = 0;
        physicalToLogicalIDs.reserve(nodeSet.size());
        logicalToPhysicalIDs.reserve(nodeSet.size());

        for (Region<1>::StreakIterator i = nodeSet.beginStreak(); i != nodeSet.endStreak(); ++i) {
            for (int j = i->origin.x(); j != i->endX; ++j) {
                logicalToPhysicalIDs << std::make_pair(j, physicalID);
                physicalToLogicalIDs << j;
                ++physicalID;
            }
        }

        delegate.resize(CoordBox<1>(Coord<1>(0), Coord<1>(nodeSet.size())));
    }

    // fixme: 1. finish GridBase interface with internal ID translation (via logicalToPhysicalIDs)
    // fixme: 2. ID translation is slow, but acceptable for IO. not acceptable for updates.
    //           probably acceptable for ghost zone communication.

    // fixme: 3. allow steppers, simulators to obtain a remapped
    // region from a grid (e.g. remapRegion(), needs to be in
    // gridBase) and add accessors that work with these remapped regions. can probably be inside specialized updatefunctor

    inline
    void setWeights(std::size_t matrixID, const std::map<Coord<2>, WeightType>& matrix)
    {
        std::map<int, int> rowLenghts;
        for (typename std::map<Coord<2>, WeightType>::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
            ++rowLenghts[i->first.x()];
        }

        typedef std::vector<IntPair> RowLengthVec;
        RowLengthVec reorderedRowLengths;
        reorderedRowLengths.reserve(nodeSet.size());

        for (Region<1>::StreakIterator i = nodeSet.beginStreak(); i != nodeSet.endStreak(); ++i) {
            for (int j = i->origin.x(); j != i->endX; ++j) {
                reorderedRowLengths << std::make_pair(j, rowLenghts[j]);
            }
        }

        for (RowLengthVec::iterator i = reorderedRowLengths.begin(); i != reorderedRowLengths.end(); ) {
            RowLengthVec::iterator nextStop = std::min(i + SIGMA, reorderedRowLengths.end());

            std::stable_sort(i, nextStop, [](const IntPair& a, const IntPair& b) {
                    return a.second > b.second;
                });

            i = nextStop;
        }

        std::vector<IntPair> newLogicalToPhysicalIDs;
        std::vector<int> newPhysicalToLogicalIDs;
        newLogicalToPhysicalIDs.reserve(nodeSet.size());
        newPhysicalToLogicalIDs.reserve(nodeSet.size());

        for (std::size_t i = 0; i < reorderedRowLengths.size(); ++i) {
            int logicalID = reorderedRowLengths[i].first;
            newLogicalToPhysicalIDs << std::make_pair(logicalID, i);
            newPhysicalToLogicalIDs << logicalID;
        }

        std::sort(newLogicalToPhysicalIDs.begin(), newLogicalToPhysicalIDs.end(), [](const IntPair& a, const IntPair& b){
                return a.first < b.first;
            });

        reorderDelegateGrid(std::move(newLogicalToPhysicalIDs), std::move(newPhysicalToLogicalIDs));
    }

    /**
     * The extent of this grid class is defined by its node set (given
     * in the c-tor) and the edge weights. Resize doesn't make sense
     * in this context.
     */
    virtual void resize(const CoordBox<DIM>&)
    {
        throw std::logic_error("Resize not supported ReorderingUnstructuredGrid");
    }

    virtual void set(const Coord<DIM>& coord, const CellType& cell)
    {
        set(coord.x(), cell);
    }

    virtual void set(const Streak<DIM>& streak, const CellType *cells)
    {
        int index = 0;
        for (int i = streak.origin.x(); i != streak.endX; ++i) {
            set(i, cells[index]);
            ++index;
        }
    }

    virtual CellType get(const Coord<DIM>& coord) const
    {
        return get(coord.x());
    }

    virtual void get(const Streak<DIM>& streak, CellType *cells) const
    {
        int index = 0;
        for (int i = streak.origin.x(); i != streak.endX; ++i) {
            cells[index] = get(i);
            ++index;
        }
    }

    virtual void setEdge(const CellType& cell)
    {
        delegate.setEdge(cell);
    }

    virtual const CellType& getEdge() const
    {
        return delegate.getEdge();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return nodeSet.boundingBox();
    }

    virtual Region<DIM> boundingRegion() const
    {
        return nodeSet;
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
    DELEGATE_GRID delegate;
    Region<1> nodeSet;
    std::vector<IntPair> logicalToPhysicalIDs;
    std::vector<int> physicalToLogicalIDs;

    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CellType>& selector,
        const Region<DIM>& region) const
    {
        // fixme
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CellType>& selector,
        const Region<DIM>& region)
    {
        // fixme
    }

    void reorderDelegateGrid(std::vector<IntPair>&& newLogicalToPhysicalIDs, std::vector<int>&& newPhysicalToLogicalIDs)
    {
        logicalToPhysicalIDs = std::move(newLogicalToPhysicalIDs);
        physicalToLogicalIDs = std::move(newPhysicalToLogicalIDs);
    }

    inline
    CellType get(int logicalID) const
    {
        std::vector<IntPair>::const_iterator pos = std::lower_bound(
            logicalToPhysicalIDs.begin(), logicalToPhysicalIDs.end(), logicalID,
            [](const IntPair& a, const int logicalID) {
                return a.first < logicalID;
            });
        if ((pos == logicalToPhysicalIDs.end() || (pos->first != logicalID))) {
            return delegate.getEdge();
        }

        return delegate.get(Coord<1>(pos->second));
    }

    inline
    void set(int logicalID, const CellType& cell)
    {
        std::vector<IntPair>::const_iterator pos = std::lower_bound(
            logicalToPhysicalIDs.begin(), logicalToPhysicalIDs.end(), logicalID,
            [](const IntPair& a, const int logicalID) {
                return a.first < logicalID;
            });
        if ((pos == logicalToPhysicalIDs.end() || (pos->first != logicalID))) {
            delegate.setEdge(cell);
        }

        delegate.set(Coord<1>(pos->second), cell);
    }
};

}

#endif
#endif
