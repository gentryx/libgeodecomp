#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/reorderingunstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iterator>
#include <utility>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
    Iterator(const Matrix& matrix, unsigned startIndex) :
        matrix(matrix),
        index(startIndex)
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
    unsigned index;
};

}

/**
 * Simple neighborhood for UnstructuredGrid. This is used as hoodOld
 * in update() * or updateLineX(). This class does not * perform
 * coordinate remapping from logical to physical IDs, hence ID *
 * reordering is strongly suggested (e.g. via *
 * ReorderingUnstructuredGrid). This class is also used for scalar *
 * updates in the vectorized case, which is why GRID is a template *
 * parameter (think UnstructuredSoAGrid).
 *
 * Usage:
 *  for (const auto& i: hoodOld.weights()) {
 *    const CELL& cell  = hoodOld[i.first];
 *    VALUE_TYPE weight = i.second;
 *  }
 */
template<typename CELL, std::size_t MATRICES,
         typename VALUE_TYPE, int C, int SIGMA>
class UnstructuredNeighborhood
{
public:
    using Iterator = UnstructuredNeighborhoodHelpers::Iterator<VALUE_TYPE, C, SIGMA>;
    using Grid = ReorderingUnstructuredGrid<UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> >;

    inline
    UnstructuredNeighborhood(const Grid& grid, long startX) :
        grid(grid),
        xOffset(startX),
        currentChunk(static_cast<unsigned>(startX / C)),
        chunkOffset(startX % C),
        currentMatrixID(0)
    {}

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
    UnstructuredNeighborhood& weights(std::size_t matrixID = 0)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator begin() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        return Iterator(matrix, static_cast<unsigned>(index));
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        index += C * matrix.rowLengthVec()[static_cast<std::size_t>(xOffset)];
        return Iterator(matrix, static_cast<unsigned>(index));
    }

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

private:
    const Grid& grid;            /**< old grid */
    long xOffset;                /**< initial offset for updateLineX function */
    unsigned currentChunk;       /**< current chunk */
    int chunkOffset;             /**< offset inside current chunk: 0 <= x < C */
    std::size_t currentMatrixID; /**< current id for matrices */

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
};

}

#endif
#endif
