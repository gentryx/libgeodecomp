#ifndef _UNSTRUCTUREDNEIGHBORHOOD_H_
#define _UNSTRUCTUREDNEIGHBORHOOD_H_

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
    UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& grid;
    long long xOffset;          /**< initial offset for updateLineX function */
    int currentChunk;           /**< current chunk */
    int chunkOffset;            /**< offset inside current chunk: 0 <= x < C */

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
    inline explicit
    UnstructuredNeighborhood(UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& _grid,
                             long long startX) :
        grid(_grid), xOffset(startX), currentChunk(startX / C), chunkOffset(startX % C)
    {}

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

    inline
    CELL& operator[](int index)
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
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator+=(int value)
    {
        updateIndices(value);
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator-=(int value)
    {
        updateIndices(-value);
        return *this;
    }

    inline
    const long& index() const { return xOffset; }

    inline
    long& index() { return xOffset; }

    inline
    std::vector<std::pair<int, VALUE_TYPE> > weights() const
    {
        // FIXME: this is only the neighborhood for matrices[0]
        return weights(0);
    }

    inline
    std::vector<std::pair<int, VALUE_TYPE> > weights(std::size_t matrixID) const
    {
        std::vector<std::pair<int, VALUE_TYPE> > neighbors;

        // preallocate some memory: reduces memory allocations via emplace_back
        neighbors.reserve(20);

        // in bounds?
        if (!grid.boundingBox().inBounds(Coord<1>(xOffset))) {
            // FIXME: what id to return for edgeCell?
            neighbors.emplace_back(-1, static_cast<VALUE_TYPE>(-1));
            return neighbors;
        }

        // get actual neighborhood
        const auto& matrix = grid.getAdjacency(matrixID);
        int index = matrix.chunkOffsetVec()[currentChunk] + chunkOffset;
        for (int element = 0; element < matrix.rowLengthVec()[xOffset]; ++element, index += C)
            neighbors.emplace_back(matrix.columnVec()[index], matrix.valuesVec()[index]);

        return neighbors;
    }
};

}

#endif /* _UNSTRUCTUREDNEIGHBORHOOD_H_ */
