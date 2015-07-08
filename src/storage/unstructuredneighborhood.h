#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
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

    inline void operator++()
    {
        index += C;
    }

    inline bool operator==(const Iterator& other) const
    {
        // matrix is ignored, since in general it's not useful to compare iterators
        // pointing to different matrices
        return index == other.index;
    }

    inline bool operator!=(const Iterator& other) const
    {
        return !(*this == other);
    }

    inline const std::pair<int, VALUE_TYPE> operator*() const
    {
        return std::make_pair(matrix.columnVec()[index],
                              matrix.valuesVec()[index]);
    }

private:
    const Matrix& matrix;
    int index;
};

/**
 * Empty dummy class.
 */
template<typename CELL, typename GRID, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1,
         bool SORT = false>
class UnstructuredNeighborhoodBase
{};

/**
 * Base class for UnstructuredNeighborhoods. There are two implementations:
 * One NeighborHood corrects the sorting given by the SELL matrix. This one
 * does, the one below does not.
 * Moreover this class is also used for scalar updates in vectorized case.
 * This is why the GRID is a template parameter.
 */
template<typename CELL, typename GRID, std::size_t MATRICES,
         typename VALUE_TYPE, int C, int SIGMA>
class UnstructuredNeighborhoodBase<CELL, GRID, MATRICES, VALUE_TYPE, C, SIGMA, true>
{
public:
    using Grid = GRID;
    using Iterator = UnstructuredNeighborhoodHelpers::Iterator<VALUE_TYPE, C, SIGMA>;

    inline
    UnstructuredNeighborhoodBase(const Grid& grid, long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(0),
        chunkOffset(0)
    {}

    inline
    UnstructuredNeighborhoodBase& operator++()
    {
        ++xOffset;
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
    UnstructuredNeighborhoodBase& weights(const std::size_t matrixID)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator begin()
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        currentChunk = matrix.realRowToSortedVec()[xOffset] / C;
        chunkOffset  = matrix.realRowToSortedVec()[xOffset] % C;
        int index    = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator(matrix, index);
    }

    inline
    const Iterator end()
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        const int realRow = matrix.realRowToSortedVec()[xOffset];
        index += C * matrix.rowLengthVec()[realRow];
        return Iterator(matrix, index);
    }

protected:
    const Grid& grid;           /**< old grid */
    long xOffset;               /**< initial offset for updateLineX function */
    int currentChunk;           /**< current chunk */
    int chunkOffset;            /**< offset inside current chunk: 0 <= x < C */
    int currentMatrixID;        /**< current id for matrices */
};

/**
 * Same as above, except SORT = false which is faster.
 */
template<typename CELL, typename GRID, std::size_t MATRICES,
         typename VALUE_TYPE, int C, int SIGMA>
class UnstructuredNeighborhoodBase<CELL, GRID, MATRICES, VALUE_TYPE, C, SIGMA, false>
{
public:
    using Grid = GRID;
    using Iterator = UnstructuredNeighborhoodHelpers::Iterator<VALUE_TYPE, C, SIGMA>;

    inline
    UnstructuredNeighborhoodBase(const Grid& grid, long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(startX / C),
        chunkOffset(startX % C)
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
    const long& index() const { return xOffset; }

    inline
    long& index() { return xOffset; }

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
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator(matrix, index);
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
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
        public UnstructuredNeighborhoodHelpers::
        UnstructuredNeighborhoodBase<CELL, UnstructuredGrid<CELL, MATRICES,
                                                            VALUE_TYPE, C, SIGMA>,
                                     MATRICES, VALUE_TYPE, C, SIGMA, true>
{
public:
    using Grid = UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>;
    using UnstructuredNeighborhoodHelpers::
    UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, SIGMA, true>::grid;

    inline
    UnstructuredNeighborhood(const Grid& grid, long startX) :
        UnstructuredNeighborhoodHelpers::
        UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, SIGMA, true>(grid, startX)
    {}

    const CELL& operator[](int index) const
    {
        return grid[index];
    }
};

/**
 * Same as above (see doc).
 */
template<typename CELL, std::size_t MATRICES,
         typename VALUE_TYPE, int C>
class UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, 1> :
        public UnstructuredNeighborhoodHelpers::
        UnstructuredNeighborhoodBase<CELL, UnstructuredGrid<CELL, MATRICES,
                                                            VALUE_TYPE, C, 1>,
                                     MATRICES, VALUE_TYPE, C, 1, false>
{
public:
    using Grid = UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, 1>;
    using UnstructuredNeighborhoodHelpers::
    UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, 1, false>::grid;

    inline
    UnstructuredNeighborhood(const Grid& grid, long startX) :
        UnstructuredNeighborhoodHelpers::
        UnstructuredNeighborhoodBase<CELL, Grid, MATRICES, VALUE_TYPE, C, 1, false>(grid, startX)
    {}

    const CELL& operator[](int index) const
    {
        return grid[index];
    }
};

/**
 * Simple neighborhood which is used for hoodNew in updateLineX().
 * Provides access to cells via an identifier which is returned by
 * hoodOld (see Iterator classe above).
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class CellIDNeighborhood
{
public:
    using Grid = UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>;

    inline explicit
    CellIDNeighborhood(Grid& grid) :
        grid(grid)
    {}

    inline
    CELL& operator[](int index)
    {
        return grid[index];
    }

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

private:
    Grid& grid;                 /**< new grid */
};

}

#endif
#endif
