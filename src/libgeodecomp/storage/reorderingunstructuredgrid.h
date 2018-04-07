#ifndef LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_REORDERINGUNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <algorithm>
#include <libgeodecomp/storage/serializationbuffer.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

class SparseMatrixVectorMultiplication;

namespace LibGeoDecomp {

namespace ReorderingUnstructuredGridHelpers {

typedef std::pair<int, int> IntPair;

inline
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

/**
 * Helper class which converts logical coordinates to physical ones
 * (i.e. those that are actually used to address memory).
 */
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

/**
 * Type switch
 */
template<typename T>
class Selector;

/**
 * see above
 */
template<>
class Selector<APITraits::TrueType>
{
public:
    typedef ReorderingRegionIterator<3> Value;
};

/**
 * see above
 */
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
    template<typename CELL, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA>
    friend class UnstructuredNeighborhood;
    template<typename CELL>
    friend class UnstructuredUpdateFunctor;
    friend class ReorderingUnstructuredGridTest;
    friend class UnstructuredNeighborhoodTest;
    friend class UnstructuredTestCellTest;
    friend class ::SparseMatrixVectorMultiplication;

    typedef typename DELEGATE_GRID::CellType CellType;
    typedef typename DELEGATE_GRID::SparseMatrix SparseMatrix;
    typedef typename DELEGATE_GRID::StorageType StorageType;
    typedef typename DELEGATE_GRID::WeightType WeightType;
    typedef typename APITraits::SelectSoA<CellType>::Value SoAFlag;
    typedef typename SerializationBuffer<CellType>::BufferType BufferType;
    typedef typename ReorderingUnstructuredGridHelpers::Selector<SoAFlag>::Value ReorderingRegionIterator;

    typedef std::pair<int, int> IntPair;

    using GridBase<CellType, 1>::saveRegion;
    using GridBase<CellType, 1>::loadRegion;

    const static int DIM = 1;
    const static int SIGMA = DELEGATE_GRID::SIGMA;
    const static int C = DELEGATE_GRID::C;

    template<typename NODE_SET_TYPE = Region<1> >
    explicit ReorderingUnstructuredGrid(
        const NODE_SET_TYPE& nodeSet = Region<1>(),
        const CellType& defaultElement = CellType(),
        const CellType& edgeElement = CellType(),
        const Coord<1>& topologicalDimensions = Coord<1>()) :
        nodeSet(nodeSet)
    {
        int physicalID = 0;
        physicalToLogicalIDs.reserve(nodeSet.size());
        logicalToPhysicalIDs.reserve(nodeSet.size());

        for (typename NODE_SET_TYPE::StreakIterator i = nodeSet.beginStreak(); i != nodeSet.endStreak(); ++i) {
            for (int j = i->origin.x(); j != i->endX; ++j) {
                logicalToPhysicalIDs << std::make_pair(j, physicalID);
                physicalToLogicalIDs << j;
                ++physicalID;
            }
        }

        CoordBox<1> delegateBox(Coord<1>(0), Coord<1>(static_cast<int>(nodeSet.size())));
        delegate = DELEGATE_GRID(delegateBox, defaultElement, edgeElement);
    }

    /**
     * Return a pointer to the underlying data storage. Use with care!
     */
    inline
    StorageType *data()
    {
        return delegate.data();
    }

    /**
     * Return a const pointer to the underlying data storage. Use with
     * care!
     */
    inline
    const StorageType *data() const
    {
        return delegate.data();
    }

    /**
     * Set edge weights. This function also triggers the remapping of
     * the internal cell IDs.
     */
    inline
    void setWeights(std::size_t matrixID, const SparseMatrix& matrix)
    {
        std::map<int, int> rowLengths;
        Region<1> mask;
        for (typename SparseMatrix::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
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
			std::size_t currentIndex = i - reorderedRowLengths.begin();
			std::size_t nextIndex = (std::min)(currentIndex + SIGMA, reorderedRowLengths.size());
			RowLengthVec::iterator nextStop = reorderedRowLengths.begin() + nextIndex;
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
        SparseMatrix newMatrix;

        for (typename SparseMatrix::const_iterator i = matrix.begin(); i != matrix.end(); ++i) {
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

            newMatrix << std::make_pair(Coord<2>(id1, id2), i->second);
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
    virtual Region<1> remapRegion(const Region<1>& region) const
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

    /**
     * Expands a Region so that all chunks it touches are filled up.
     * This can be used by an UpdateFunktor to reduce the amount of
     * conditionals but comes at the expense of additional
     * computations.
     */
    virtual Region<1> expandChunksInRemappedRegion(const Region<1>& region)
    {
        Region<1> ret;
        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<1> streak = *i;
            streak.origin.x() = streak.origin.x() / C * C;
            streak.endX = (streak.endX + C - 1) / C * C;

            ret << streak;
        }

        return ret;
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor) const
    {
        delegate.callback(functor);
    }

    template<typename FUNCTOR>
    void callback(ReorderingUnstructuredGrid *newGrid, FUNCTOR functor) const
    {
        delegate.callback(&newGrid->delegate, functor);
    }

    inline
    const SellCSigmaSparseMatrixContainer<WeightType, C, SIGMA>& getWeights(const std::size_t matrixID) const
    {
        return delegate.getWeights(matrixID);
    }

    inline
    SellCSigmaSparseMatrixContainer<WeightType, C, SIGMA>& getWeights(const std::size_t matrixID)
    {
        return delegate.getWeights(matrixID);
    }

private:
    DELEGATE_GRID delegate;
    Region<1> nodeSet;
    std::vector<IntPair> logicalToPhysicalIDs;
    std::vector<int> physicalToLogicalIDs;

    /**
     * This operator is private as it gives access access to the
     * underlying delegate without mapping the id to the physical ID.
     * This interface is required by update functor and neighborhood
     * objects for performance reasons, but a potential source of
     * errors if used unintentionally. Hence users must be "friends"
     * with this class.
     */
    CellType& operator[](int index)
    {
        return delegate[index];
    }

    /**
     * This operator is private as it gives access access to the
     * underlying delegate without mapping the id to the physical ID.
     * This interface is required by update functor and neighborhood
     * objects for performance reasons, but a potential source of
     * errors if used unintentionally. Hence users must be "friends"
     * with this class.
     */
    const CellType& operator[](int index) const
    {
        return delegate[index];
    }

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
