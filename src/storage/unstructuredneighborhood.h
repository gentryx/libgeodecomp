#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/reorderingunstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

#include <iterator>
#include <utility>
#include <vector>

namespace LibGeoDecomp {

namespace UnstructuredNeighborhoodHelpers {

/**
 * Used for iterating over neighboring cells.
 */
template<typename VALUE_TYPE, int C, int SIGMA>
class Iterator : public std::iterator<std::forward_iterator_tag,
                                      const std::pair<int, VALUE_TYPE> >
{
public:
    using Matrix = SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>;

    inline
    Iterator(const Matrix& matrix, int startIndex) :
        matrix(matrix), index(startIndex)
    {}

    inline
    void operator++()
    {
        index += C;
    }

    inline
    bool operator==(const Iterator& other) const
    {
        // matrix is ignored, since in general it's not useful to compare iterators
        // pointing to different matrices
        return index == other.index;
    }

    inline
    bool operator!=(const Iterator& other) const
    {
        return !(*this == other);
    }

    inline
    const Iterator& operator*() const
    {
        return *this;
    }

    inline
    int first() const
    {
        return matrix.columnVec()[index];
    }

    inline
    VALUE_TYPE second() const
    {
        return matrix.valuesVec()[index];
    }


private:
    const Matrix& matrix;
    int index;
};

/**
 * Base class for UnstructuredNeighborhoods. This class does not
 * perform coordinate remapping from logical to physical IDs, hence ID
 * reordering is strongly suggested (e.g. via
 * ReorderingUnstructuredGrid). This class is also used for scalar
 * updates in the vectorized case, which is why GRID is a template
 * parameter (think UnstructuredSoAGrid).
 */
template<typename CELL, typename GRID, std::size_t MATRICES,
         typename VALUE_TYPE, int C, int SIGMA>
class UnstructuredNeighborhoodBase
{
public:
    using Grid = GRID;
    using Iterator = UnstructuredNeighborhoodHelpers::Iterator<VALUE_TYPE, C, SIGMA>;

    inline
    UnstructuredNeighborhoodBase(const Grid& grid, long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(startX / C),
        chunkOffset(startX % C),
        currentMatrixID(0)
    {}

    inline
    UnstructuredNeighborhoodBase& operator++()
    {
        updateIndices(1);
        return *this;
    }

    inline
    UnstructuredNeighborhoodBase operator++(int)
    {
        UnstructuredNeighborhoodBase tmp(*this);
        operator++();
        return tmp;
    }

    inline
    void operator+=(long i)
    {
        xOffset += i;
    }

    inline
    long index() const
    {
        return xOffset;
    }

    inline
    long index()
    {
        return xOffset;
    }

    inline
    UnstructuredNeighborhoodBase& weights()
    {
        // default neighborhood is for matrix 0
        return weights(0);
    }

    inline
    UnstructuredNeighborhoodBase& weights(std::size_t matrixID)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator begin() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator(matrix, index);
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        index += C * matrix.rowLengthVec()[xOffset];
        return Iterator(matrix, index);
    }

protected:
    /**
     * If xOffset is changed, the current chunk and chunkOffset
     * may change. This function updates the internal data structures
     * accordingly.
     *
     * @param difference amount which is added to xOffset
     */
    inline
    void updateIndices(int difference)
    {
        xOffset += difference;
        const int newChunkOffset = chunkOffset + difference;

        // update chunk and offset, if necessary
        if (newChunkOffset >= C) {
            ++currentChunk;
            chunkOffset = 0;
            return;
        }

        chunkOffset += difference;
    }

    const Grid& grid;           /**< old grid */
    long xOffset;               /**< initial offset for updateLineX function */
    int currentChunk;           /**< current chunk */
    int chunkOffset;            /**< offset inside current chunk: 0 <= x < C */
    int currentMatrixID;        /**< current id for matrices */
};

}

/**
 * Simple neighborhood for UnstructuredGrid. This is used as hoodOld in update()
 * or updateLineX(). There are also two implementations: if SIGMA = 1 -> SORT = false,
 * else SIGMA > 1 -> SORT = true.
 *
 * Usage:
 *  for (const auto& i: hoodOld.weights()) {
 *    const CELL& cell  = hoodOld[i.first];
 *    VALUE_TYPE weight = i.second;
 *  }
 */
template<typename CELL, std::size_t MATRICES,
         typename VALUE_TYPE, int C, int SIGMA>
class UnstructuredNeighborhood :
        public UnstructuredNeighborhoodHelpers::UnstructuredNeighborhoodBase<
    CELL, ReorderingUnstructuredGrid<UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> >, MATRICES, VALUE_TYPE, C, SIGMA>
{
public:
    using Grid = ReorderingUnstructuredGrid<UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> >;
    using UnstructuredNeighborhoodHelpers::
    UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, SIGMA>::grid;

    inline
    UnstructuredNeighborhood(const Grid& grid, long startX) :
        UnstructuredNeighborhoodHelpers::
        UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, SIGMA>(grid, startX)
    {}

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }
};

}

#endif
#endif
