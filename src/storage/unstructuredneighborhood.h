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

/**
 * Simple neighborhood for UnstructuredGrid.
 *
 * Usage:
 *  for (const auto& i: hoodOld.weights()) {
 *    const CELL& cell  = hoodOld[i.first];
 *    VALUE_TYPE weight = i.second;
 *  }
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredNeighborhood
{
private:
    typedef UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> Grid;
    const Grid& grid;
    long long xOffset;          /**< initial offset for updateLineX function */
    int currentChunk;           /**< current chunk */
    int chunkOffset;            /**< offset inside current chunk: 0 <= x < C */
    int currentMatrixID;        /**< current id for matrices */

    /**
     * If xOffset is changed, the current chunk and chunkOffset
     * may change. This function updates the internal data structures
     * accordingly.
     *
     * @param difference amount which is added or subtracted from xOffset
     */
    void updateIndices(int difference)
    {
        xOffset += difference;
        const int newChunkOffset = chunkOffset + difference;

        // update chunk and offset, if necessary
        if (newChunkOffset < 0) {
            --currentChunk;
            chunkOffset = C - 1;
            return;
        } else if (newChunkOffset >= C) {
            ++currentChunk;
            chunkOffset = 0;
            return;
        }

        chunkOffset += difference;
    }
public:

    /**
     * Used for iterating over neighboring cells.
     */
    template<typename O_VALUE_TYPE, int O_C, int O_SIGMA>
    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const std::pair<int, O_VALUE_TYPE> >
    {
    private:
        typedef SellCSigmaSparseMatrixContainer<O_VALUE_TYPE, O_C, O_SIGMA> Matrix;
        const Matrix& matrix;
        int index;
        std::pair<int, O_VALUE_TYPE> currentPair;
    public:
        inline
        Iterator(const Matrix& matrix, int startIndex) :
            matrix(matrix), index(startIndex),
            currentPair(std::make_pair(matrix.columnVec()[index],
                                       matrix.valuesVec()[index]))
        {}

        inline void operator++()
        {
            index += O_C;
            std::get<0>(currentPair) = matrix.columnVec()[index];
            std::get<1>(currentPair) = matrix.valuesVec()[index];
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

        inline const std::pair<int, VALUE_TYPE>& operator*() const
        {
            return currentPair;
        }

        inline const std::pair<int, VALUE_TYPE> *operator->() const
        {
            return &currentPair;
        }
    };

    inline
    UnstructuredNeighborhood(const Grid& grid, long long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(startX / C),
        chunkOffset(startX % C)
    {}

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator++()
    {
        updateIndices(1);
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator++(int)
    {
        UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        operator++();
        return tmp;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator--()
    {
        updateIndices(-1);
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator--(int)
    {
        UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        operator--();
        return tmp;
    }

    inline
    const long long& index() const { return xOffset; }

    inline
    long long& index() { return xOffset; }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& weights()
    {
        // default neighborhood is for matrix 0
        return weights(0);
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& weights(std::size_t matrixID)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator<VALUE_TYPE, C, SIGMA> begin() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator<VALUE_TYPE, C, SIGMA>(matrix, index);
    }

    inline
    const Iterator<VALUE_TYPE, C, SIGMA> end() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        index += C * matrix.rowLengthVec()[xOffset];
        return Iterator<VALUE_TYPE, C, SIGMA>(matrix, index);
    }
};

/**
 * Simple neighborhood which is used for hoodNew in updateLineX().
 * Provides access to cells via an identifier.
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class CellIDNeighborhood
{
private:
    typedef UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> Grid;
    Grid& grid;
public:
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
};

}

#endif
#endif
