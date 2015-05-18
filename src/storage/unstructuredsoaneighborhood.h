#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/flat_array.hpp>
#include <libflatarray/short_vec.hpp>
#include <libflatarray/soa_accessor.hpp>

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>

#include <iterator>
#include <utility>

namespace LibGeoDecomp {

namespace UnstructuredSoANeighborhoodHelpers {

template<typename CELL, typename VALUE_TYPE>
class GetMemberPointer
{
private:
    VALUE_TYPE **memberPointer;
public:
    inline GetMemberPointer(VALUE_TYPE **ptr) :
        memberPointer(ptr)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor)
    {
        // save member pointer
        *memberPointer = static_cast<VALUE_TYPE *>(&accessor.sum());
    }
};

}

/**
 * Neighborhood using vectorization for UnstructuredSoAGrid.
 * Weights() returns a pair of LFA short_vec classes to update
 * current chunk of SELL-C-q.
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredSoANeighborhood
{
private:
    typedef UnstructuredSoAGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> Grid;
    typedef LibFlatArray::short_vec<VALUE_TYPE, C> ShortVec;
    typedef LibFlatArray::short_vec<VALUE_TYPE, C> ShortVecScalar;
    typedef std::pair<ShortVec, ShortVec> IteratorPair;
    const Grid& grid;
    int currentChunk;           /**< current chunk */
    int currentMatrixID;        /**< current id for matrices */
    VALUE_TYPE *sumPtr;         /**< pointer to sum member of CELL */

public:
    /**
     * This iterator returns LFA short_vec classes needed to update
     * the current chunk. Itherators goes through:
     *    j = 0; j < chunkLength[currentChunk]; ++j
     */
    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const IteratorPair>
    {
    private:
        typedef SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA> Matrix;
        const Matrix& matrix;   /**< matrix to use */
        int offset;             /**< where are we right now?  */
        VALUE_TYPE *sumPtr;     /**< pointer to sum data */

    public:
        inline
        Iterator(const Matrix& matrix, int offset, VALUE_TYPE *memberPtr) :
            matrix(matrix), offset(offset), sumPtr(memberPtr)
        {}

        inline
        void operator++()
        {
            offset += C;
        }

        inline
        bool operator==(const Iterator& other) const
        {
            // offset is indicater here
            return offset == other.offset;
        }

        inline
        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        inline
        const IteratorPair operator*() const
        {
            // load next vectors
            // get indices for rhs gather -> corresponds to column vector
            const unsigned *indices = reinterpret_cast<const unsigned *>(matrix.columnVec().data()) + offset;
            ShortVec cellValues;
            cellValues.gather(sumPtr, const_cast<unsigned *>(indices));
            ShortVec matrixValues = matrix.valuesVec().data() + offset;
            return std::make_pair(cellValues, matrixValues);
        }
    };

    inline
    UnstructuredSoANeighborhood(const Grid& grid, long long startX) :
        grid(grid),
        currentChunk(startX / C)
    {
        // save member pointer
        grid.callback(UnstructuredSoANeighborhoodHelpers::
                      GetMemberPointer<CELL, VALUE_TYPE>(&sumPtr));
    }

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator++()
    {
        ++currentChunk;
        return *this;
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator++(int)
    {
        UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        operator++();
        return tmp;
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator--()
    {
        --currentChunk;
        return *this;
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator--(int)
    {
        UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        operator--();
        return tmp;
    }

    inline
    const int& index() const
    {
        return currentChunk;
    }

    inline
    int& index()
    {
        return currentChunk;
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& weights()
    {
        // default neighborhood is for matrix 0
        return weights(0);
    }

    inline
    UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& weights(std::size_t matrixID)
    {
        currentMatrixID = matrixID;

        return *this;
    }

    inline
    Iterator begin() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        return Iterator(matrix,
                        matrix.chunkOffsetVec()[currentChunk],
                        sumPtr);
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        // FIXME
        return Iterator(matrix,
                        matrix.chunkOffsetVec()[currentChunk + 1],
                        sumPtr);
    }
};

/**
 * Neighborhood which is used for hoodNew in updateLineX().
 * Provides access to member pointers of the new grid.
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredSoANeighborhoodNew
{
private:
    typedef UnstructuredSoAGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> Grid;
    typedef LibFlatArray::short_vec<VALUE_TYPE, C> ShortVec;
    Grid& grid;
    VALUE_TYPE *sumPtr;
public:
    inline explicit
    UnstructuredSoANeighborhoodNew(Grid& grid) :
        grid(grid)
    {
        grid.callback(UnstructuredSoANeighborhoodHelpers::
                      GetMemberPointer<CELL, VALUE_TYPE>(&sumPtr));
    }

    inline
    UnstructuredSoANeighborhoodNew& operator[](int index)
    {
        // setup sum vector
        sum = sumPtr + index * C;
        return *this;
    }

    inline
    const UnstructuredSoANeighborhoodNew& operator[](int index) const
    {
        // setup sum vector
        sum = sumPtr + index * C;
        return *this;
    }

    // public cell members as ShortVecs
    ShortVec sum;
};

}

#endif
#endif
