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
    SortItem(std::size_t length, std::size_t index) :
        rowLength(length),
        rowIndex(index)
    {}
    std::size_t rowLength;
    std::size_t rowIndex;
};

/**
 * Helper class to initialize the sell container from an adjacency matrix.
 * This is a class and not a method, because there are two different implementations,
 * one for SIGMA = 1 and one for SIGMA > 1.
 */
template<typename VALUETYPE, std::size_t C, std::size_t SIGMA>
class InitFromMatrix
{
public:
    using SellContainer = SellCSigmaSparseMatrixContainer<VALUETYPE, C, SIGMA>;
    using Matrix = std::vector<std::pair<Coord<2>, VALUETYPE> >;

    void operator()(SellContainer *container, const Matrix& matrix) const
    {
        std::vector<int> rowLengthCopy;

        // calculate size for arrays
        const std::size_t matrixRows = container->dimension;
        const std::size_t numberOfChunks = (matrixRows - 1) / C + 1;
        const std::size_t numberOfSigmas = (matrixRows - 1) / SIGMA + 1;
        const std::size_t rowsPadded = numberOfChunks * C;
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
        chunkLength.resize(numberOfChunks + 0);
        rowLength.resize(rowsPadded);
        rowLengthCopy.resize(rowsPadded);
        realRowToSorted.reserve(rowsPadded);
        chunkRowToReal.resize(rowsPadded);

        // get row lengths
        std::fill(begin(rowLength), end(rowLength), 0);
        for (const auto& pair: matrix) {
            ++rowLength[static_cast<unsigned>(pair.first.x())];
        }

        // map sorting scope
        for (std::size_t nSigma = 0; nSigma < numberOfSigmas; ++nSigma) {
            const std::size_t numberOfRows = (std::min)(SIGMA, rowsPadded - nSigma * SIGMA);
            std::vector<SortItem> lengths(numberOfRows);
            for (std::size_t i = 0; i < numberOfRows; ++i) {
                const std::size_t row = nSigma * SIGMA + i;
                lengths[i] = SortItem(static_cast<std::size_t>(rowLength[row]), row);
            }
            std::stable_sort(begin(lengths), end(lengths),
                             [] (const SortItem& a, const SortItem& b) -> bool
                             { return a.rowLength > b.rowLength; });

            for (unsigned i = 0; i < numberOfRows; ++i) {
                unsigned newID = nSigma * SIGMA + i;
                int newIndex = static_cast<int>(lengths[i].rowIndex);
                chunkRowToReal[newID] = newIndex;
                realRowToSorted.push_back(std::make_pair(newIndex, newID));
                rowLengthCopy[newID] = static_cast<int>(lengths[i].rowLength);
            }
        }

        std::stable_sort(realRowToSorted.begin(), realRowToSorted.end(),
                         [] (const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool
                         { return a.first < b.first; });

        // save chunk lengths and offsets
        chunkOffset[0] = 0;
        for (std::size_t nChunk = 0; nChunk < numberOfChunks; ++nChunk) {
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
        chunkOffset[numberOfChunks] =
            chunkOffset[numberOfChunks - 1] +
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

            std::vector<std::pair<int, int> >::iterator iter = std::lower_bound(
                realRowToSorted.begin(), realRowToSorted.end(), pair.first.x(),
                [](const std::pair<int, int>& a, const int id) {
                    return a.first < id;
                });

            if (iter == realRowToSorted.end()) {
                throw std::logic_error("ID not found");
            }

            const int chunk = iter->second / C;
            const int row   = iter->second % C;
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
    typedef std::vector<std::pair<Coord<2>, VALUETYPE> > SparseMatrix;
    using AlignedValueVector = std::vector<VALUETYPE, LibFlatArray::aligned_allocator<VALUETYPE, 64> >;
    using AlignedIntVector   = std::vector<int, LibFlatArray::aligned_allocator<int, 64> >;

    friend SellHelpers::InitFromMatrix<VALUETYPE, C, SIGMA>;
    friend class ReorderingUnstructuredGridTest;

    explicit
    SellCSigmaSparseMatrixContainer(const int N = 0) :
        values(),
        column(),
        rowLength(static_cast<std::size_t>(N), 0),
        chunkLength(static_cast<std::size_t>((N - 1) / C + 1), 0),
        chunkOffset(static_cast<std::size_t>((N - 1) / C + 2), 0),
        dimension(N)
    {
        static_assert(C >= 1, "C should be greater or equal to 1!");
        static_assert(SIGMA >= 1, "SIGMA should be greater or equal to 1!");
    }

    /**
     * Returns all neighbors of a given ID (i.e. all nodes to which a
     * node (row) has edges leading to, or in other words: the indices
     * of all non-zero entries in the matrix' row). Useful for
     * debugging and IO, not efficient for use in kernels.
     */
    std::vector<std::pair<int, VALUETYPE> > getRow(int const row) const
    {
        std::vector< std::pair<int, VALUETYPE> > vec;
        const int chunk(row / C);
        const int offset(row % C);
        int index = chunkOffset[static_cast<unsigned>(chunk)] + offset;

        for (int element = 0; element < rowLength[row]; ++element, index += C) {
            vec.push_back(std::pair<int, VALUETYPE>(
                              column[static_cast<unsigned>(index)],
                              values[static_cast<unsigned>(index)]));
        }

        return vec;
    }

    /**
     * This method can be used, if this container should be initialized from a
     * _complete_ matrix. Matrix is represented as map, key is Coord<2> which contains
     * (row, column). value_type of map contains the actual value.
     */
    void initFromMatrix(const SparseMatrix& matrix)
    {
        SparseMatrix sortedMatrix = matrix;
        std::sort(sortedMatrix.begin(), sortedMatrix.end(), [](const std::pair<Coord<2>, VALUETYPE>& a, const std::pair<Coord<2>, VALUETYPE>& b){
                return a.first < b.first;
            });
        SellHelpers::InitFromMatrix<VALUETYPE, C, SIGMA>()(this, sortedMatrix);
    }

    inline bool operator==(const SellCSigmaSparseMatrixContainer& other) const
    {
        return ((dimension   == other.dimension)  &&
                (values      == other.values)     &&
                (column      == other.column)     &&
                (chunkLength == other.chunkLength));
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

    inline const std::vector<std::pair<int, int> >& realRowToSortedVec() const
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
    AlignedIntVector column;
    std::vector<int> rowLength;       // = Non Zero Entres in Row
    std::vector<int> chunkLength;     // = Max rowLength in Chunk
    std::vector<int> chunkOffset;     // COffset[i+1]=COffset[i]+CLength[i]*C
    std::vector<std::pair<int, int> > realRowToSorted; // mapping between rows and real rows, used for SIGMA
    std::vector<int> chunkRowToReal;  // and the other way around
    std::size_t dimension;              // = N
};

}

#endif
#endif
