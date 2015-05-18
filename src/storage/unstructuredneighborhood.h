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
 * Simple neighborhood for UnstructuredGrid. This implementation
 * takes SIGMA into account. This is slower than using SIGMA = 1,
 * since the chunk and offset are computed by an indirection.
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

public:
    /**
     * Used for iterating over neighboring cells.
     */
    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const std::pair<int, VALUE_TYPE> >
    {
    private:
        typedef SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA> Matrix;
        const Matrix& matrix;
        int index;
    public:
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
    };

    inline
    UnstructuredNeighborhood(const Grid& grid, long long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(0),
        chunkOffset(0)
    {}

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

    inline
    UnstructuredNeighborhood& operator++()
    {
        ++xOffset;
        return *this;
    }

    inline
    UnstructuredNeighborhood operator++(int)
    {
        UnstructuredNeighborhood tmp(*this);
        operator++();
        return tmp;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator--()
    {
        --xOffset;
        return *this;
    }

    inline
    UnstructuredNeighborhood operator--(int)
    {
        UnstructuredNeighborhood tmp(*this);
        operator--();
        return tmp;
    }

    inline
    const long long& index() const { return xOffset; }

    inline
    long long& index() { return xOffset; }

    inline
    UnstructuredNeighborhood& weights()
    {
        // default neighborhood is for matrix 0
        return weights(0);
    }

    inline
    UnstructuredNeighborhood& weights(std::size_t matrixID)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator begin()
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        currentChunk = matrix.rowIndicesVec()[xOffset] / C;
        chunkOffset  = matrix.rowIndicesVec()[xOffset] % C;
        int index    = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator(matrix, index);
    }

    inline
    const Iterator end()
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        index    += C * matrix.rowLengthVec()[xOffset];
        return Iterator(matrix, index);
    }
};

/**
 * Simple neighborhood for UnstructuredGrid. SIGMA = 1.
 *
 * Usage:
 *  for (const auto& i: hoodOld.weights()) {
 *    const CELL& cell  = hoodOld[i.first];
 *    VALUE_TYPE weight = i.second;
 *  }
 */
template<typename CELL, std::size_t MATRICES, typename VALUE_TYPE, int C>
class UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, 1>
{
private:
    typedef UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, 1> Grid;
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
    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const std::pair<int, VALUE_TYPE> >
    {
    private:
        typedef SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, 1> Matrix;
        const Matrix& matrix;
        int index;
    public:
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
    UnstructuredNeighborhood& operator++()
    {
        updateIndices(1);
        return *this;
    }

    inline
    UnstructuredNeighborhood operator++(int)
    {
        UnstructuredNeighborhood tmp(*this);
        operator++();
        return tmp;
    }

    inline
    UnstructuredNeighborhood& operator--()
    {
        updateIndices(-1);
        return *this;
    }

    inline
    UnstructuredNeighborhood operator--(int)
    {
        UnstructuredNeighborhood tmp(*this);
        operator--();
        return tmp;
    }

    inline
    const long long& index() const { return xOffset; }

    inline
    long long& index() { return xOffset; }

    inline
    UnstructuredNeighborhood& weights()
    {
        // default neighborhood is for matrix 0
        return weights(0);
    }

    inline
    UnstructuredNeighborhood& weights(std::size_t matrixID)
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
