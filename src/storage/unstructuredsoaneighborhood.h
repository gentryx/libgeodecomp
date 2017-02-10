#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/flat_array.hpp>
#include <libflatarray/soa_accessor.hpp>

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>

#include <iterator>
#include <utility>

namespace LibGeoDecomp {

namespace UnstructuredSoANeighborhoodHelpers
{

// fixme: doc
template<typename HOOD>
class WrappedNeighborhood
{
public:
    typedef typename HOOD::ScalarIterator Iterator;

    inline
    WrappedNeighborhood(HOOD& hood) :
        hood(hood)
    {}

    inline
    ~WrappedNeighborhood()
    {
        hood.intraChunkOffset = (hood.intraChunkOffset + 1) % HOOD::ARITY;
    }

    inline
    Iterator begin()
    {
        return hood.beginScalar();
    }

    inline
    Iterator end()
    {
        return hood.endScalar();
    }

private:
    HOOD& hood;
};

/**
 * fixme: doc
 */
template<typename HOOD, int CHUNK_SIZE, int ITERATOR_ARITY>
class NeighborhoodWrapper;

template<typename HOOD>
class NeighborhoodWrapper<HOOD, 1, 1>
{
public:
    typedef HOOD& Value;

    inline
    Value& operator()(HOOD& hood)
    {
        return hood;
    }
};

template<typename HOOD, int CHUNK_SIZE>
class NeighborhoodWrapper<HOOD, CHUNK_SIZE, 1>
{
public:
    typedef WrappedNeighborhood<HOOD> Value;

    inline
    Value operator()(HOOD& hood)
    {
        return Value(hood);
    }
};

template<typename HOOD, int CHUNK_SIZE>
class NeighborhoodWrapper<HOOD, CHUNK_SIZE, CHUNK_SIZE>
{
public:
    typedef HOOD& Value;

    inline
    Value& operator()(HOOD& hood)
    {
        return hood;
    }
};

}

/**
 * Neighborhood providing pointers for vectorization of UnstructuredSoAGrid.
 * weights(id) returns a pair of two pointers. One points to the array where
 * the indices for gather are stored and the seconds points the matrix values.
 * Both pointers can be used to load LFA short_vec classes accordingly.
 *
 * DIM_X/Y/Z refer to the grid's storage dimensions, INDEX is a fixed
 * offset for the soa_accessor, MATRICES is the number of adjacency
 * matrices (equals 1 for most applications), VALUE_TYPE is the type
 * of the edge weights, C refers to the chunk size and SIGMA is the
 * sorting scope used by the SELL-C-Sigma container.
 */
template<
    typename GRID_TYPE,
    typename CELL,
    long DIM_X,
    long DIM_Y,
    long DIM_Z,
    long INDEX,
    std::size_t MATRICES = 1,
    typename VALUE_TYPE = double,
    int C = 64,
    int SIGMA = 1>
class UnstructuredSoANeighborhood
{
public:
    template<typename HOOD>
    friend class UnstructuredSoANeighborhoodHelpers::WrappedNeighborhood;

    static const int ARITY = C;

    using IteratorPair = std::pair<const int*, const VALUE_TYPE*>;
    using SoAAccessor = LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>;
    using ConstSoAAccessor = LibFlatArray::const_soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>;

    /**
     * This iterator returns objects/values needed to update
     * the current chunk. Iterator consists of a pair: indices reference
     * and matrix values reference.
     */
    class Iterator : public std::iterator<std::forward_iterator_tag, const IteratorPair>
    {
    public:
        using Matrix = SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>;

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
            return offset == other.offset;
        }

        inline
        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        inline const Iterator& operator*() const
        {
            return *this;
        }

        inline
        const int *first() const
        {
            return &matrix.columnVec()[offset];
        }

        inline
        const VALUE_TYPE *second() const
        {
            return &matrix.valuesVec()[offset];
        }

    private:
        const Matrix& matrix;   // Which matrix to use?
        int offset;             // In which chunk are we right now?
    };

    /**
     * These iterators are used to traverse parts of a chunk, which
     * may be necessary during loop peeling at the begin or end of a
     * Streak if the Streak isn't aligned on the chunk size C.
     */
    class ScalarIterator : public std::iterator<std::forward_iterator_tag, const IteratorPair>
    {
    public:
        using Matrix = SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>;

        inline
        ScalarIterator(const Matrix& matrix, int offset, int scalarOffset) :
            matrix(matrix),
            offset(offset),
            scalarOffset(scalarOffset)
        {}

        inline
        void operator++()
        {
            offset += C;
        }

        inline
        bool operator==(const ScalarIterator& other) const
        {
            return offset == other.offset;
        }

        inline
        bool operator!=(const ScalarIterator& other) const
        {
            return !(*this == other);
        }

        inline const ScalarIterator& operator*() const
        {
            return *this;
        }

        inline
        const int *first() const
        {
            return &matrix.columnVec()[offset] + scalarOffset;
        }

        inline
        const VALUE_TYPE *second() const
        {
            return &matrix.valuesVec()[offset] + scalarOffset;
        }

    private:
        const Matrix& matrix;   // Which matrix to use?
        int offset;             // In which chunk are we right now?
        int scalarOffset;       // Our offset within the chunk
    };

    inline
    UnstructuredSoANeighborhood(const SoAAccessor& acc, const GRID_TYPE& grid, long startX, int intraChunkOffset = 0) :
        grid(grid),
        currentChunk(startX / C),
        currentMatrixID(0),
        intraChunkOffset(intraChunkOffset),
        accessor(acc)
    {}

    inline
    UnstructuredSoANeighborhood& operator++()
    {
        ++currentChunk;
        return *this;
    }

    inline
    int index() const
    {
        return currentChunk;
    }

    inline
    int index()
    {
        return currentChunk;
    }

    inline
    UnstructuredSoANeighborhood& weights(std::size_t matrixID = 0)
    {
        currentMatrixID = matrixID;
        return *this;
    }

    template<typename SHORT_VEC>
    inline
    typename UnstructuredSoANeighborhoodHelpers::NeighborhoodWrapper<UnstructuredSoANeighborhood, C, SHORT_VEC::ARITY>::Value weights(SHORT_VEC, std::size_t matrixID = 0)
    {
        currentMatrixID = matrixID;
        typedef typename UnstructuredSoANeighborhoodHelpers::NeighborhoodWrapper<UnstructuredSoANeighborhood, C, SHORT_VEC::ARITY> Fixme;
        return Fixme()(*this);
    }

    inline
    Iterator begin() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        return Iterator(matrix, matrix.chunkOffsetVec()[currentChunk]);
    }

    inline
    const Iterator end() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        return Iterator(matrix, matrix.chunkOffsetVec()[currentChunk + 1]);
    }

    inline
    ScalarIterator beginScalar() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        return ScalarIterator(matrix, matrix.chunkOffsetVec()[currentChunk], intraChunkOffset);
    }

    inline
    const ScalarIterator endScalar() const
    {
        const auto& matrix = grid.getWeights(currentMatrixID);
        return ScalarIterator(matrix, matrix.chunkOffsetVec()[currentChunk + 1], intraChunkOffset);
    }

    inline
    const SoAAccessor *operator->() const
    {
        return &accessor;
    }

    inline
    const ConstSoAAccessor operator[](int offset) const
    {
        return ConstSoAAccessor(accessor.data(), accessor.index() + offset);
    }

private:
    // fixme: get rid of grid reference, only reference weight vector
    const GRID_TYPE& grid;       /**< old grid */
    /**
     * index of current chunk in weights container (stored in
     * SELL-C-Sigma format)
     */
    int currentChunk;
    /**
     * logical ID of current adjacency matrix (virtually always 0)
     */
    int currentMatrixID;
    /**
     * offset within a chunk, useful for loop peeling if a Streak
     * doesn't start and/or end on a chunkboundary:
     */
    int intraChunkOffset;
    /**
     * Struct-of-Arrays accessor to old grid:
     */
    const SoAAccessor& accessor;
};

template<
    typename GRID_TYPE,
    typename CELL,
    long DIM_X,
    long DIM_Y,
    long DIM_Z,
    long INDEX,
    std::size_t MATRICES,
    typename VALUE_TYPE,
    int C,
    int SIGMA>
const int UnstructuredSoANeighborhood<GRID_TYPE, CELL, DIM_X, DIM_Y, DIM_Z, INDEX, MATRICES, VALUE_TYPE, C, SIGMA>::ARITY;

}

#endif
#endif
