#ifndef LIBGEODECOMP_STORAGE_SELLCSIGMASPARSEMATRIXCONTAINER_H
#define LIBGEODECOMP_STORAGE_SELLCSIGMASPARSEMATRIXCONTAINER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/aligned_allocator.hpp>
#include <libgeodecomp/geometry/coord.h>

#include <map>
#include <vector>
#include <utility>
#include <assert.h>
#include <stdexcept>
#include <algorithm>

#include <iostream>

namespace LibGeoDecomp {

template<typename VALUETYPE, int C, int SIGMA>
class SellCSigmaSparseMatrixContainer;

namespace SellHelpers {

/**
 * Helper struct used for sorting.
 */
struct SortItem
{
public:
    SortItem() :
        rowLength(0), rowIndex(0)
    {}
    SortItem(int length, int index) :
        rowLength(length), rowIndex(index)
    {}
    int rowLength;
    int rowIndex;
};

/**
 * Helper class to initialize the sell container from an adjacency matrix.
 * This is a class and not a method, because there are two different implementations,
 * one for SIGMA = 1 and one for SIGMA > 1.
 */
template<typename VALUETYPE, int C, int SIGMA>
class InitFromMatrix
{
public:
    using SellContainer = SellCSigmaSparseMatrixContainer<VALUETYPE, C, SIGMA>;
    using Matrix = std::map<Coord<2>, VALUETYPE>;

    void operator()(SellContainer *container, const Matrix& matrix) const
    {
        std::vector<int> rowLengthCopy;

        // calculate size for arrays
        const int matrixRows = container->dimension;
        const int numberOfChunks = (matrixRows - 1) / C + 1;
        const int numberOfSigmas = (matrixRows - 1) / SIGMA + 1;
        const int rowsPadded = numberOfChunks * C;
        int numberOfValues = 0;

        // save references to sell data structures
        auto& chunkOffset     = container->chunkOffset;
        auto& chunkLength     = container->chunkLength;
        auto& rowLength       = container->rowLength;
        auto& realRowToSorted = container->realRowToSorted;
        auto& chunkRowToReal  = container->chunkRowToReal;
        auto& values          = container->values;
        auto& column          = container->column;

        // allocate memory
        chunkOffset.resize(numberOfChunks + 1);
        chunkLength.resize(numberOfChunks);
        rowLength.resize(rowsPadded);
        rowLengthCopy.resize(rowsPadded);
        realRowToSorted.resize(rowsPadded);
        chunkRowToReal.resize(rowsPadded);

        // get row lengths
        std::fill(begin(rowLength), end(rowLength), 0);
        for (const auto& pair: matrix) {
            ++rowLength[pair.first.x()];
        }

        // map sorting scope
        for (int nSigma = 0; nSigma < numberOfSigmas; ++nSigma) {
            const int numberOfRows = std::min(SIGMA, rowsPadded - nSigma * SIGMA);
            std::vector<SortItem> lengths(numberOfRows);
            for (int i = 0; i < numberOfRows; ++i) {
                const int row = nSigma * SIGMA + i;
                lengths[i] = SortItem(rowLength[row], row);
            }
            std::stable_sort(begin(lengths), end(lengths),
                             [] (const SortItem& a, const SortItem& b) -> bool
                             { return a.rowLength > b.rowLength; });
            for (int i = 0; i < numberOfRows; ++i) {
                chunkRowToReal[nSigma * SIGMA + i]   = lengths[i].rowIndex;
                realRowToSorted[lengths[i].rowIndex] = nSigma * SIGMA + i;
                rowLengthCopy[nSigma * SIGMA + i] = lengths[i].rowLength;
            }
        }

        // save chunk lengths and offsets
        chunkOffset[0] = 0;
        for (int nChunk = 0; nChunk < numberOfChunks; ++nChunk) {
            std::vector<int> lengths(C);
            for (auto i = 0u; i < lengths.size(); ++i) {
                lengths[i] = rowLength[chunkRowToReal[nChunk * C + i]];
            }
            chunkLength[nChunk] = *std::max_element(begin(lengths), end(lengths));
            if (nChunk > 0) {
                chunkOffset[nChunk] = chunkOffset[nChunk - 1] + chunkLength[nChunk - 1] * C;
            }
            numberOfValues += chunkLength[nChunk] * C;
        }
        chunkOffset[numberOfChunks] = chunkOffset[numberOfChunks - 1] +
            chunkLength[numberOfChunks - 1] * C;

        // save values
        rowLength = std::move(rowLengthCopy);
        values.resize(numberOfValues);
        column.resize(numberOfValues);
        std::fill(begin(values), end(values), 0);
        std::fill(begin(column), end(column), 0);
        int currentRow = 0;
        int index = 0;
        for (const auto& pair: matrix) {
            if (pair.first.x() != currentRow) {
                currentRow = pair.first.x();
                index = 0;
            }
            const int chunk = realRowToSorted[pair.first.x()] / C;
            const int row   = realRowToSorted[pair.first.x()] % C;
            const int start = chunkOffset[chunk];
            const int idx   = start + index * C + row;
            values[idx]     = pair.second;
            column[idx]     = pair.first.y();
            ++index;
        }
    }
};

/**
 * See doc above.
 */
template<typename VALUETYPE, int C>
class InitFromMatrix<VALUETYPE, C, 1>
{
public:
    using SellContainer = SellCSigmaSparseMatrixContainer<VALUETYPE, C, 1>;
    using Matrix = std::map<Coord<2>, VALUETYPE>;

    void operator()(SellContainer *container, const Matrix& matrix) const
    {
        // calculate size for arrays
        const int matrixRows = container->dimension;
        const int numberOfChunks = (matrixRows - 1) / C + 1;
        const int rowsPadded = numberOfChunks * C;
        int numberOfValues = 0;

        // save references to sell data structures
        auto& chunkOffset     = container->chunkOffset;
        auto& chunkLength     = container->chunkLength;
        auto& rowLength       = container->rowLength;
        auto& realRowToSorted = container->realRowToSorted;
        auto& chunkRowToReal  = container->chunkRowToReal;
        auto& values          = container->values;
        auto& column          = container->column;

        // allocate memory
        chunkOffset.resize(numberOfChunks + 1);
        chunkLength.resize(numberOfChunks);
        rowLength.resize(rowsPadded);

        // get row lengths
        std::fill(begin(rowLength), end(rowLength), 0);
        for (const auto& pair: matrix) {
            ++rowLength[pair.first.x()];
        }

        // save chunk lengths and offsets
        chunkOffset[0] = 0;
        for (int nChunk = 0; nChunk < numberOfChunks; ++nChunk) {
            chunkLength[nChunk] = *std::max_element(rowLength.begin() + nChunk * C,
                                                    rowLength.begin() + (nChunk + 1) * C);
            if (nChunk > 0) {
                chunkOffset[nChunk] = chunkOffset[nChunk - 1] + chunkLength[nChunk - 1] * C;
            }
            numberOfValues += chunkLength[nChunk] * C;
        }
        chunkOffset[numberOfChunks] = chunkOffset[numberOfChunks - 1] +
            chunkLength[numberOfChunks - 1] * C;

        // save values
        values.resize(numberOfValues);
        column.resize(numberOfValues);
        std::fill(begin(values), end(values), 0);
        std::fill(begin(column), end(column), 0);
        int currentRow = 0;
        int index = 0;
        for (const auto& pair: matrix) {
            if (pair.first.x() != currentRow) {
                currentRow = pair.first.x();
                index = 0;
            }
            const int chunk = pair.first.x() / C;
            const int row   = pair.first.x() % C;
            const int start = chunkOffset[chunk];
            const int idx   = start + index * C + row;
            values[idx]     = pair.second;
            column[idx]     = pair.first.y();
            ++index;
        }
    }
};

}

/**
 * This class represents an container which uses an efficient
 * storage layout for sparse matrices.
 *
 * See: http://arxiv.org/abs/1307.6209
 */
template<typename VALUETYPE, int C = 1, int SIGMA = 1>
class SellCSigmaSparseMatrixContainer
{
public:
    using AlignedValueVector = std::vector<VALUETYPE, LibFlatArray::aligned_allocator<VALUETYPE, 64> >;
    using AlignedIntVector   = std::vector<int, LibFlatArray::aligned_allocator<int, 64> >;

    friend SellHelpers::InitFromMatrix<VALUETYPE, C, SIGMA>;

    explicit
    SellCSigmaSparseMatrixContainer(const int N = 0) :
        values(),
        column(),
        rowLength(N, 0),
        chunkLength((N-1)/C + 1, 0),
        chunkOffset((N-1)/C + 2, 0),
        dimension(N)
    {
        static_assert(C >= 1, "C should be greater or equal to 1!");
        static_assert(SIGMA >= 1, "SIGMA should be greater or equal to 1!");
    }

    // lhs = A   x rhs
    // tmp = val x b
    void matVecMul(std::vector<VALUETYPE>& lhs, std::vector<VALUETYPE>& rhs)
    {
        if (lhs.size() != rhs.size() || lhs.size() != dimension) {
            throw std::invalid_argument("lhs and rhs must be of size N");
        }

        // loop over chunks     TODO parallel omp
        for (std::size_t chunk = 0; chunk < chunkLength.size(); ++chunk) {
            int offs = chunkOffset[chunk];
            VALUETYPE tmp[C];

            // init tmp                     TODO vectorize
            for (int row = 0; row<C; ++row) {
                tmp[row] = lhs[chunk*C + row];
            }

            // loop over columns in chunk
            for (int col = 0; col < chunkLength[chunk]; ++col) {

                // loop over rows in chunks TODO vectorize
                for (int row = 0; row < C; ++row) {
                    VALUETYPE val = values[offs];
                    int columnINDEX = column[offs++];
                    // note: val might be zero due to padding
                    VALUETYPE b = rhs[columnINDEX];
                    tmp[row] += val * b;
                }
            }

            // store tmp                     TODO vectorize
            for (int row = 0; row < C; ++row) {
                lhs[chunk*C + row] = tmp[row];
            }
        }

    }

    // fixme: is this mainly used for constructing the neighborhood in UnstructuredGrid::getNeighborhood. drop this code once we have an efficient neighborhood-object for UnstructuredGrid
    std::vector<std::pair<int, VALUETYPE> > getRow(int const row) const
    {
        std::vector< std::pair<int, VALUETYPE> > vec;
        int const chunk (row/C);
        int const offset (row%C);
        int index = chunkOffset[chunk] + offset;

        for (int element = 0;
             element < rowLength[row];
             ++element, index += C) {
            vec.push_back(std::pair<int, VALUETYPE>(column[index], values[index]));
        }

        return vec;
    }

    /**
     * This method can be used, if this container should be initialized from a
     * _complete_ matrix. Matrix is represented as map, key is Coord<2> which contains
     * (row, column). value_type of map contains the actual value.
     */
    void initFromMatrix(const std::map<Coord<2>, VALUETYPE>& matrix)
    {
        SellHelpers::InitFromMatrix<VALUETYPE, C, SIGMA>()(this, matrix);
    }

    inline bool operator==(const SellCSigmaSparseMatrixContainer& other) const
    {
        return (dimension   == other.dimension  &&
                values      == other.values     &&
                column      == other.column     &&
                chunkLength == other.chunkLength);
    }

    template<int O_C, int O_SIGMA>
    inline bool operator==(const SellCSigmaSparseMatrixContainer<VALUETYPE, O_C, O_SIGMA>& other) const
    {
        if (dimension != other.dimension) {
            return false;
        }

        for (std::size_t i=0; i<dimension; ++i) {
            if (getRow(i) != other.getRow(i)) {
                return false;
            }
        }

        return true;
    }

    template<typename OTHER>
    inline bool operator!=(const OTHER& other) const
    {
        return !(*this == other);
    }

    inline const AlignedValueVector& valuesVec() const
    {
        return values;
    }

    inline const AlignedIntVector& columnVec() const
    {
        return column;
    }

    inline const std::vector<int>& rowLengthVec() const
    {
        return rowLength;
    }

    inline const std::vector<int>& chunkLengthVec() const
    {
        return chunkLength;
    }

    inline const std::vector<int>& chunkOffsetVec() const
    {
        return chunkOffset;
    }

    inline const std::vector<int>& realRowToSortedVec() const
    {
        return realRowToSorted;
    }

    inline const std::vector<int>& chunkRowToRealVec() const
    {
        return chunkRowToReal;
    }

    inline std::size_t dim() const
    {
        return dimension;
    }

private:
    AlignedValueVector values;
    AlignedIntVector   column;
    std::vector<int>   rowLength;       // = Non Zero Entres in Row
    std::vector<int>   chunkLength;     // = Max rowLength in Chunk
    std::vector<int>   chunkOffset;     // COffset[i+1]=COffset[i]+CLength[i]*C
    std::vector<int>   realRowToSorted; // mapping between rows and real rows, used for SIGMA
    std::vector<int>   chunkRowToReal;  // and the other way around
    std::size_t dimension;              // = N
};

}

#endif
#endif
