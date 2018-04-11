#ifndef LIBGEODECOMP_IO_SELLSORTINGWRITER_H
#define LIBGEODECOMP_IO_SELLSORTINGWRITER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/selector.h>

// Kill some warnings in system headers:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4996)
#endif

#include <string>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstring>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

namespace SellSortingWriterHelpers {

/**
 * Helper class which sorts the actual UnstructuredSoAGrid members
 * regarding to used SELL matrix.
 */
template<typename CELL, typename VALUE_TYPE, int C, int SIGMA>
class SortMember
{
public:
    inline
    SortMember(const Selector<CELL>& selector,
               const SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>& matrix,
               bool forward) :
        selector(selector),
        matrix(matrix),
        forward(forward)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor)
    {
        const auto& rowsVec = forward ? matrix.realRowToSortedVec() : matrix.chunkRowToRealVec();
        const std::size_t size = rowsVec.size(); // -> corresponds to rowsPadded
        char *data = accessor.access_member(selector.sizeOfMember(), selector.offset());
        std::vector<char> copy(size * selector.sizeOfMember());
        std::memcpy(copy.data(), data, size * selector.sizeOfMember());

        // sort
        for (std::size_t row = 0; row < size; ++row) {
            std::size_t realRow = rowsVec[row];
            // e.g. copy 8 bytes for double
            std::memcpy(data + row * selector.sizeOfMember(),
                        copy.data() + realRow * selector.sizeOfMember(),
                        selector.sizeOfMember());
        }
    }

private:
    const Selector<CELL>& selector;
    const SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>& matrix;
    bool forward;
};

}

/**
 * This writer works as a proxy writer. If the user use unstructured grids
 * and vectorization (SoA memory layout) the output has to be sorted according
 * to the used SELL matrix. This writer sorts the output grid and just calls
 * the real writer.
 */
template<typename CELL, typename WRITER>
class SellSortingWriter : public Clonable<Writer<CELL>, SellSortingWriter<CELL, WRITER> >
{
public:
    using GridType = typename Writer<CELL>::GridType;
    using Topology = typename APITraits::SelectTopology<CELL>::Value;
    static const auto DIM = Topology::DIM;
    static const auto MATRICES = APITraits::SelectSellMatrices<CELL>::VALUE;
    static const auto C = APITraits::SelectSellC<CELL>::VALUE;
    static const auto SIGMA = APITraits::SelectSellSigma<CELL>::VALUE;
    using Writer<CELL>::period;
    using Writer<CELL>::prefix;
    using ValueType = typename APITraits::SelectSellType<CELL>::Value;
    using SoAGrid = UnstructuredSoAGrid<CELL, MATRICES, ValueType, C, SIGMA>;

    template<typename MEMBER>
    SellSortingWriter(WRITER *proxy,
                      std::size_t matrixID,
                      const std::string& prefix,
                      MEMBER CELL:: *memberPointer,
                      const unsigned period = 1) :
        Clonable<Writer<CELL>, SellSortingWriter<CELL, WRITER> >(prefix, period),
        delegate(proxy),
        selector(memberPointer, "unused name"),
        matrixID(matrixID)
    {
        if (SIGMA <= 1) {
            throw std::logic_error("The SortingWriter makes only sense to use with a SIGMA greater 1.");
        }
        if (delegate == nullptr) {
            throw std::invalid_argument("Writer pointer is NULL.");
        }
    }

    virtual ~SellSortingWriter()
    {
        delete delegate;
    }

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        // sort forward for output
        sort(grid, true);
        delegate->stepFinished(grid, step, event);
        // sort back for further computations
        sort(grid, false);
    }

private:
    void sort(const GridType& grid, bool forward)
    {
        const auto *soaGrid = dynamic_cast<const SoAGrid *>(&grid);
        if (soaGrid == nullptr) {
            throw std::logic_error("SellSortingWriter can only be used with UnstructuredSoAGrid. "
                                   "Did you forget to specify HasSoA apitrait?");
        }

        const auto& matrix = soaGrid->getWeights(matrixID);
        // fixme: we'll need to rework this api at some later point of time as a
        //        writer should treat the grid as read-only'
        soaGrid->callback(SellSortingWriterHelpers::
                          SortMember<CELL, ValueType, C, SIGMA>(selector, matrix, forward));
    }

    WRITER *delegate;
    Selector<CELL> selector;
    std::size_t matrixID;
};

}

#endif
#endif
