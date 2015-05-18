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

// FIXME: this classes should be automatically generated according
//        to Cell's members
/**
 * Helper class to get the pointer to Cell's
 * sum variable in SoA grid.
 */
template<typename CELL, typename VALUE_TYPE>
class GetSumPointer
{
private:
    VALUE_TYPE **memberPointer;
public:
    inline GetSumPointer(VALUE_TYPE **ptr) :
        memberPointer(ptr)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor)
    {
        // FIXME: index?
        // save member pointer
        *memberPointer = static_cast<VALUE_TYPE *>(&accessor.sum());
    }
};

/**
 * Helper class to get the pointer to Cell's
 * value variable in SoA grid.
 */
template<typename CELL, typename VALUE_TYPE>
class GetValuePointer
{
private:
    VALUE_TYPE **memberPointer;
public:
    inline GetValuePointer(VALUE_TYPE **ptr) :
        memberPointer(ptr)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor)
    {
        // FIXME: index?
        // save member pointer
        *memberPointer = static_cast<VALUE_TYPE *>(&accessor.value());
    }
};

}

/**
 * Neighborhood using vectorization for UnstructuredSoAGrid.
 * Weights() returns a pair of two pointers. One points to
 * the array where the indices for gather are stored and the
 * seconds points the matrix values. Both pointers can be used
 * to load LFA short_vec classes accordingly.
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredSoANeighborhood
{
private:
    typedef UnstructuredSoAGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> Grid;
    typedef LibFlatArray::short_vec<VALUE_TYPE, C> ShortVec;
    typedef LibFlatArray::short_vec<VALUE_TYPE, C> ShortVecScalar;
    typedef std::pair<const unsigned *, const VALUE_TYPE *> IteratorPair;
    const Grid& grid;
    int currentChunk;           /**< current chunk */
    int currentMatrixID;        /**< current id for matrices */

public:
    /**
     * This iterator returns objects/values needed to update
     * the current chunk. Iterator consists of a pair: indices pointer
     * and matrix values pointer.
     */
    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const IteratorPair>
    {
    private:
        typedef SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA> Matrix;
        const Matrix& matrix;   /**< matrix to use */
        int offset;             /**< Where are we right now inside chunk?  */

    public:
        inline
        Iterator(const Matrix& matrix, int offset) :
            matrix(matrix), offset(offset)
        {}

        inline
        void operator++()
        {
            offset += C;
        }

        inline
        bool operator==(const Iterator& other) const
        {
            // offset is indicator here
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
            // load indices and matrix values pointers
            const VALUE_TYPE *weights = matrix.valuesVec().data() + offset;
            const unsigned *indices =
                reinterpret_cast<const unsigned *>(matrix.columnVec().data()) + offset;
            return std::make_pair(indices, weights);
        }
    };

    inline
    UnstructuredSoANeighborhood(const Grid& grid, long long startX) :
        grid(grid),
        currentChunk(startX / C)
    {
        // save member pointer
        grid.callback(UnstructuredSoANeighborhoodHelpers::
                      GetValuePointer<CELL, VALUE_TYPE>(&valuePtr));
    }

    inline
    UnstructuredSoANeighborhood& operator++()
    {
        ++currentChunk;
        return *this;
    }

    inline
    UnstructuredSoANeighborhood operator++(int)
    {
        UnstructuredSoANeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        operator++();
        return tmp;
    }

    inline
    UnstructuredSoANeighborhood& operator--()
    {
        --currentChunk;
        return *this;
    }

    inline
    UnstructuredSoANeighborhood operator--(int)
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
    UnstructuredSoANeighborhood& weights()
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
        return Iterator(matrix, matrix.chunkOffsetVec()[currentChunk]);
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getAdjacency(currentMatrixID);
        return Iterator(matrix, matrix.chunkOffsetVec()[currentChunk + 1]);
    }

    // public cell members
    VALUE_TYPE *valuePtr;
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
public:
    inline explicit
    UnstructuredSoANeighborhoodNew(Grid& grid) :
        grid(grid)
    {
        grid.callback(UnstructuredSoANeighborhoodHelpers::
                      GetSumPointer<CELL, VALUE_TYPE>(&sumPtr));
    }

    // public cell members
    VALUE_TYPE *sumPtr;
};

}

#endif
#endif
