#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libflatarray/cuda_array.hpp>

#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/cudautil.h>
#include <libgeodecomp/storage/serializationbuffer.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

namespace CUDASoAGridHelpers {

template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
__global__
void set_kernel(
    const CELL *edgeCell,
    const CELL *innerCell,
    char *target,
    long dimX,
    long dimY,
    long dimZ,
    long edgeRadiiX,
    long edgeRadiiY,
    long edgeRadiiZ,
    bool initInterior)
{
    long x = blockDim.x * blockIdx.x + threadIdx.x;
    long y = blockDim.y * blockIdx.y + threadIdx.y;
    long z = blockDim.z * blockIdx.z + threadIdx.z;

    if ((x >= dimX) ||
        (y >= dimY) ||
        (z >= dimZ)) {
        return;
    }

    typedef LibFlatArray::soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z) + INDEX;
    accessor_type accessor(target, index);

    const CELL *reference = innerCell;
    if ((x < edgeRadiiX) || (x >= (dimX - edgeRadiiX)) ||
        (y < edgeRadiiY) || (y >= (dimY - edgeRadiiY)) ||
        (z < edgeRadiiZ) || (z >= (dimZ - edgeRadiiZ))) {
        reference = edgeCell;
    } else {
        if (!initInterior) {
            return;
        }
    }

    accessor << *reference;
}

/**
 * Helper class, derived from SoAGrid's helpers.
 */
template<typename CELL, bool INIT_INTERIOR>
class SetContent
{
public:
    SetContent(
        char *data,
        const Coord<3>& gridDim,
        const Coord<3>& edgeRadii,
        const CELL& edgeCell,
        const CELL& innerCell) :
        data(data),
        gridDim(gridDim),
        edgeRadii(edgeRadii),
        edgeCellBuffer(1, edgeCell),
        innerCellBuffer(1, innerCell)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        dim3 cudaGridDim;
        dim3 cudaBlockDim;
        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, gridDim);

        set_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<cudaGridDim, cudaBlockDim>>>(
            edgeCellBuffer.data(),
            innerCellBuffer.data(),
            data,
            gridDim.x(),
            gridDim.y(),
            gridDim.z(),
            edgeRadii.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            INIT_INTERIOR);
    }

private:
    char *data;
    Coord<3> gridDim;
    Coord<3> edgeRadii;
    LibFlatArray::cuda_array<CELL> edgeCellBuffer;
    LibFlatArray::cuda_array<CELL> innerCellBuffer;
};

}

/**
 * CUDA-enabled implementation of SoAGrid, relies on
 * LibFlatArray::soa_grid.
 */
template<typename CELL,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDASoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class CUDASoAGridTest;

    const static int DIM = TOPOLOGY::DIM;

    using GridBase<CELL, DIM>::topoDimensions;
    using GridBase<CELL, DIM>::saveRegion;
    using GridBase<CELL, DIM>::loadRegion;

    /**
     * Accumulated size of all members. Note that this may be lower
     * than sizeof(CELL) as the compiler may add padding within a
     * struct/class to ensure alignment.
     */
    static const int AGGREGATED_MEMBER_SIZE =  LibFlatArray::aggregated_member_size<CELL>::VALUE;

    typedef CELL CellType;
    typedef TOPOLOGY Topology;
    typedef LibFlatArray::soa_grid<CELL, LibFlatArray::cuda_allocator<char>, true> Delegate;
    typedef typename APITraits::SelectStencil<CELL>::Value Stencil;

    explicit CUDASoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL& defaultCell = CELL(),
        const CELL& edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL, DIM>(topologicalDimensions),
        edgeRadii(calcEdgeRadii()),
        edgeCell(edgeCell),
        box(box)
    {
        // don't set edges here, but...
        resize(box, false);
        // ...init edges AND interior here in one go
        delegate.callback(
            CUDASoAGridHelpers::SetContent<CELL, true>(
                delegate.data(),
                actualDimensions,
                edgeRadii,
                edgeCell,
                defaultCell));
    }

    explicit CUDASoAGrid(
        const Region<DIM>& region,
        const CELL& defaultCell = CELL(),
        const CELL& edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL, DIM>(topologicalDimensions),
        edgeRadii(calcEdgeRadii()),
        edgeCell(edgeCell),
        box(region.boundingBox())
    {
        // don't set edges here, but...
        resize(box, false);
        // ...init edges AND interior here in one go
        delegate.callback(
            CUDASoAGridHelpers::SetContent<CELL, true>(
                delegate.data(),
                actualDimensions,
                edgeRadii,
                edgeCell,
                defaultCell));
    }

    /**
     * Return a pointer to the underlying data storage. Use with care!
     */
    inline
    char *data()
    {
        return delegate.data();
    }

    /**
     * Return a const pointer to the underlying data storage. Use with
     * care!
     */
    inline
    const char *data() const
    {
        return delegate.data();
    }

    inline void resize(const CoordBox<DIM>& newBox)
    {
        resize(newBox, true);
    }

    inline void resize(const CoordBox<DIM>& newBox, bool setEdges)
    {
        box = newBox;
        actualDimensions = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            actualDimensions[i] = newBox.dimensions[i];
        }
        actualDimensions += edgeRadii * 2;

        delegate.resize(
            actualDimensions.x(),
            actualDimensions.y(),
            actualDimensions.z());

        if (setEdges) {
            delegate.callback(
                CUDASoAGridHelpers::SetContent<CELL, false>(
                    delegate.data(),
                    actualDimensions,
                    edgeRadii,
                    edgeCell,
                    edgeCell));
        }
    }

    void set(const Coord<DIM>& absoluteCoord, const CELL& cell)
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            setEdge(cell);
            return;
        }

        delegateSet(relativeCoord, cell);
    }

    void set(const Streak<DIM>& streak, const CELL* cells)
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateSet(relativeCoord, cells, streak.length());
    }

    CELL get(const Coord<DIM>& absoluteCoord) const
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            return edgeCell;
        }

        return delegateGet(relativeCoord);
    }

    void get(const Streak<DIM>& streak, CELL *cells) const
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateGet(relativeCoord, cells, streak.length());
    }

    void setEdge(const CELL& newEdgeCell)
    {
        edgeCell = newEdgeCell;

        delegate.callback(
            CUDASoAGridHelpers::SetContent<CELL, false>(
                delegate.data(),
                actualDimensions,
                edgeRadii,
                edgeCell,
                edgeCell));
    }

    const CELL& getEdge() const
    {
        return edgeCell;
    }

    CoordBox<DIM> boundingBox() const
    {
        return box;
    }

    void saveRegion(std::vector<char> *target, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        std::size_t expectedMinimumSize = SerializationBuffer<CELL>::minimumStorageSize(region);
        if (target->size() < expectedMinimumSize) {
            throw std::logic_error(
                "target buffer too small (is " + StringOps::itoa(target->size()) +
                ", expected at least: " + StringOps::itoa(expectedMinimumSize) + ")");
        }

        SerializationBuffer<CELL>::resize(target, region);
        Coord<3> actualOffset = edgeRadii;
        for (int i = 0; i < DIM; ++i) {
            actualOffset[i] += -box.origin[i] + offset[i];
        }

        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), actualOffset);
        StreakIteratorType end(  region.endStreak(),   actualOffset);

        delegate.save(start, end, target->data(), region.size());
    }

    void loadRegion(const std::vector<char>& source, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        std::size_t expectedMinimumSize = SerializationBuffer<CELL>::minimumStorageSize(region);
        if (source.size() < expectedMinimumSize) {
            throw std::logic_error(
                "source buffer too small (is " + StringOps::itoa(source.size()) +
                ", expected at least: " + StringOps::itoa(expectedMinimumSize) + ")");
        }

        Coord<3> actualOffset = edgeRadii;
        for (int i = 0; i < DIM; ++i) {
            actualOffset[i] += -box.origin[i] + offset[i];
        }

        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), actualOffset);
        StreakIteratorType end(  region.endStreak(),   actualOffset);

        delegate.load(start, end, source.data(), region.size());
    }

protected:
    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        delegate.callback(
            SoAGridHelpers::SaveMember<CELL, DIM>(
                target,
                MemoryLocation::CUDA_DEVICE,
                targetLocation,
                selector,
                region,
                box.origin,
                edgeRadii));
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        delegate.callback(
            SoAGridHelpers::LoadMember<CELL, DIM>(
                source,
                sourceLocation,
                MemoryLocation::CUDA_DEVICE,
                selector,
                region,
                box.origin,
                edgeRadii));
    }

private:
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;

    static Coord<3> calcEdgeRadii()
    {
        return SoAGrid<CELL, Topology>::calcEdgeRadii();
    }

    CELL delegateGet(const Coord<1>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z());
    }

    CELL delegateGet(const Coord<2>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z());
    }

    CELL delegateGet(const Coord<3>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z());
    }

    void delegateGet(const Coord<1>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateGet(const Coord<2>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateGet(const Coord<3>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<1>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<2>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<3>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cell);
    }

    void delegateSet(const Coord<1>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<2>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<3>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cells,
            count);
    }
};

}

#endif
