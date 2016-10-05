#ifndef LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <algorithm>

namespace LibGeoDecomp {

namespace ReorderingUnstructuredGridHelpers {

typedef std::pair<int, int> IntPair;

std::vector<IntPair>::const_iterator mapLogicalToPhysicalID(int logicalID, const std::vector<IntPair>& logicalToPhysicalIDs)
{
    std::vector<IntPair>::const_iterator pos = std::lower_bound(
        logicalToPhysicalIDs.begin(), logicalToPhysicalIDs.end(), logicalID,
        [](const IntPair& a, const int logicalID) {
            return a.first < logicalID;
        });

    if (pos->first != logicalID) {
        return logicalToPhysicalIDs.end();
    }

    return pos;
}

template<int DIM>
class ReorderingRegionIterator
{
public:
    inline
    ReorderingRegionIterator(const Region<1>::Iterator& iter, const std::vector<IntPair>& logicalToPhysicalIDs) :
        iter(iter),
        logicalToPhysicalIDs(logicalToPhysicalIDs)
    {
        updateOrigin();
    }

    inline
    int length() const
    {
        return 1;
    }

    inline
    void operator++()
    {
        ++iter;
        updateOrigin();
    }

    inline
    const ReorderingRegionIterator *operator->() const
    {
        return this;
    }

    inline
    Streak<DIM> operator*() const
    {
        return Streak<DIM>(origin, origin.x() + 1);
    }

    Coord<DIM> origin;

    inline
    bool operator!=(const ReorderingRegionIterator& other)
    {
        return iter != other.iter;
    }

private:
    Region<1>::Iterator iter;
    const std::vector<IntPair>& logicalToPhysicalIDs;

    void updateOrigin()
    {
        std::vector<IntPair>::const_iterator i = mapLogicalToPhysicalID(iter->x(), logicalToPhysicalIDs);
        if (i != logicalToPhysicalIDs.end()) {
            origin.x() = i->second;
        }
    }
};

template<typename T>
class Selector;

template<>
class Selector<APITraits::TrueType>
{
public:
    typedef ReorderingRegionIterator<3> Value;
};

template<>
class Selector<APITraits::FalseType>
{
public:
    typedef ReorderingRegionIterator<1> Value;
};

}

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
    friend class ReorderingUnstructuredGridTest;

    typedef typename DELEGATE_GRID::CellType CellType;
    typedef typename DELEGATE_GRID::WeightType WeightType;
    typedef typename SerializationBuffer<CellType>::BufferType BufferType;
    typedef typename APITraits::SelectSoA<CellType>::Value SoAFlag;
    typedef typename ReorderingUnstructuredGridHelpers::Selector<SoAFlag>::Value ReorderingRegionIterator;


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
        std::map<int, int> rowLengths;
        Region<1> mask;
        for (typename std::map<Coord<2>, WeightType>::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
            int id = i->first.x();
            if (!nodeSet.count(Coord<1>(id))) {
                continue;
            }

            int neighborID = i->first.y();
            if ((!nodeSet.count(Coord<1>(neighborID))) || (mask.count(Coord<1>(id)))) {
                // prune nodes with missing neighbors to have 0
                // neighbors as we can safely assume they won't be
                // updated anyway.
                mask << Coord<1>(id);
                rowLengths[id] = 0;
            } else {
                ++rowLengths[id];
            }
        }

        typedef std::vector<IntPair> RowLengthVec;
        RowLengthVec reorderedRowLengths;
        reorderedRowLengths.reserve(nodeSet.size());

        for (Region<1>::StreakIterator i = nodeSet.beginStreak(); i != nodeSet.endStreak(); ++i) {
            for (int j = i->origin.x(); j != i->endX; ++j) {
                reorderedRowLengths << std::make_pair(j, rowLengths[j]);
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

        using ReorderingUnstructuredGridHelpers::mapLogicalToPhysicalID;
        std::map<Coord<2>, WeightType> newMatrix;

        for (typename std::map<Coord<2>, WeightType>::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
            int id = i->first.x();
            if (nodeSet.count(Coord<1>(id)) == 0) {
                continue;
            }

            if (mask.count(Coord<1>(id))) {
                continue;
            }

            std::vector<std::pair<int, int> >::const_iterator iter;
            iter = mapLogicalToPhysicalID(i->first.x(), logicalToPhysicalIDs);
            if (iter == logicalToPhysicalIDs.end()) {
                throw std::logic_error("unknown ID in matrix");
            }
            int id1 = iter->second;

            iter = mapLogicalToPhysicalID(i->first.y(), logicalToPhysicalIDs);
            if (iter == logicalToPhysicalIDs.end()) {
                throw std::logic_error("unknown neighbor ID in matrix");
            }
            int id2 = iter->second;

            newMatrix[Coord<2>(id1, id2)] = i->second;
        }

        delegate.setWeights(matrixID, std::move(newMatrix));
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

    virtual const Region<DIM>& boundingRegion()
    {
        return nodeSet;
    }

    virtual void saveRegion(BufferType *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        delegate.saveRegion(
            buffer,
            ReorderingRegionIterator(region.begin(), logicalToPhysicalIDs),
            ReorderingRegionIterator(region.end(), logicalToPhysicalIDs),
            region.size());
    }

    virtual void loadRegion(const BufferType& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        delegate.loadRegion(
            buffer,
            ReorderingRegionIterator(region.begin(), logicalToPhysicalIDs),
            ReorderingRegionIterator(region.end(), logicalToPhysicalIDs),
            region.size());
    }

    /**
     * Convert coordinates from Region to the internal reordering of
     * the grid so they can be used directly by an UpdateFunctor.
     */
    virtual Region<1> remapRegion(const Region<1>& region)
    {
        Region<1> ret;

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            using ReorderingUnstructuredGridHelpers::mapLogicalToPhysicalID;
            std::vector<IntPair>::const_iterator iter = mapLogicalToPhysicalID(i->x(), logicalToPhysicalIDs);

            if (iter == logicalToPhysicalIDs.end()) {
                throw std::logic_error("cannot remap Coord from Region -- Region needs to be a subset of nodeSet");
            }

            ret << Coord<1>(iter->second);
        }

        return ret;
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
        delegate.saveMemberImplementation(
            target,
            targetLocation,
            selector,
            ReorderingRegionIterator(region.begin(), logicalToPhysicalIDs),
            ReorderingRegionIterator(region.end(), logicalToPhysicalIDs));
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CellType>& selector,
        const Region<DIM>& region)
    {
        delegate.loadMemberImplementation(
            source,
            sourceLocation,
            selector,
            ReorderingRegionIterator(region.begin(), logicalToPhysicalIDs),
            ReorderingRegionIterator(region.end(), logicalToPhysicalIDs));
    }

    void reorderDelegateGrid(std::vector<IntPair>&& newLogicalToPhysicalIDs, std::vector<int>&& newPhysicalToLogicalIDs)
    {
        CoordBox<1> box(Coord<1>(), nodeSet.boundingBox().dimensions);
        DELEGATE_GRID newDelegate(box);
        for (Region<1>::Iterator i = nodeSet.begin(); i != nodeSet.end(); ++i) {
            using ReorderingUnstructuredGridHelpers::mapLogicalToPhysicalID;
            std::vector<IntPair>::const_iterator iter;

            iter = mapLogicalToPhysicalID(i->x(), newLogicalToPhysicalIDs);
            if (iter == newLogicalToPhysicalIDs.end()) {
                throw std::logic_error("ID not found in new ID map");
            }
            Coord<1> newPhysicalID(iter->second);

            iter = mapLogicalToPhysicalID(i->x(), logicalToPhysicalIDs);
            if (iter == logicalToPhysicalIDs.end()) {
                throw std::logic_error("ID not found in old ID map");
            }
            Coord<1> oldPhysicalID(iter->second);

            newDelegate.set(newPhysicalID, delegate.get(oldPhysicalID));
        }
        delegate = std::move(newDelegate);

        logicalToPhysicalIDs = std::move(newLogicalToPhysicalIDs);
        physicalToLogicalIDs = std::move(newPhysicalToLogicalIDs);
    }

    inline
    CellType get(int logicalID) const
    {
        using ReorderingUnstructuredGridHelpers::mapLogicalToPhysicalID;
        std::vector<IntPair>::const_iterator pos = mapLogicalToPhysicalID(logicalID, logicalToPhysicalIDs);

        if ((pos == logicalToPhysicalIDs.end() || (pos->first != logicalID))) {
            return delegate.getEdge();
        }

        return delegate.get(Coord<1>(pos->second));
    }

    inline
    void set(int logicalID, const CellType& cell)
    {
        using ReorderingUnstructuredGridHelpers::mapLogicalToPhysicalID;
        std::vector<IntPair>::const_iterator pos = mapLogicalToPhysicalID(logicalID, logicalToPhysicalIDs);

        if ((pos == logicalToPhysicalIDs.end() || (pos->first != logicalID))) {
            delegate.setEdge(cell);
        }

        delegate.set(Coord<1>(pos->second), cell);
    }
};

}

#endif
#endif
