#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libflatarray/cuda_array.hpp>

#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/storage/soagrid.h>

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
    long edgeRadiiZ)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if ((x >= dimX) ||
        (y >= dimY) ||
        (z >= dimZ)) {
        return;
    }

    typedef LibFlatArray::soa_accessor_light<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor_type;

    long index = accessor_type::gen_index(x, y, z) + INDEX;
    accessor_type accessor(target, index);

    const CELL *reference = innerCell;
    if ((x < edgeRadiiX) || (x >= (dimX - edgeRadiiX))) {
        reference = edgeCell;
    }
    if ((y < edgeRadiiY) || (y >= (dimY - edgeRadiiY))) {
        reference = edgeCell;
    }
    if ((z < edgeRadiiZ) || (z >= (dimZ - edgeRadiiZ))) {
        reference = edgeCell;
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
        generateLaunchConfig(&cudaGridDim, &cudaBlockDim, gridDim);

        set_kernel<CELL, DIM_X, DIM_Y, DIM_Z, INDEX><<<cudaGridDim, cudaBlockDim>>>(
            edgeCellBuffer.data(),
            innerCellBuffer.data(),
            data,
            gridDim.x(),
            gridDim.y(),
            gridDim.z(),
            edgeRadii.x(),
            edgeRadii.y(),
            edgeRadii.z());
    }

private:
    char *data;
    Coord<3> gridDim;
    Coord<3> edgeRadii;
    LibFlatArray::cuda_array<CELL> edgeCellBuffer;
    LibFlatArray::cuda_array<CELL> innerCellBuffer;

    static void generateLaunchConfig(dim3 *grid_dim, dim3 *block_dim, const Coord<3>& dim)
    {
        if (dim.y() >= 4) {
            *block_dim = dim3(128, 4, 1);
        } else {
            *block_dim = dim3(512, 1, 1);
        }

        grid_dim->x = divideAndRoundUp(dim.x(), block_dim->x);
        grid_dim->y = divideAndRoundUp(dim.y(), block_dim->y);
        grid_dim->z = divideAndRoundUp(dim.z(), block_dim->z);
    }

private:
    static int divideAndRoundUp(int i, int dividend)
    {
        int ret = i / dividend;
        if (i % dividend) {
            ret += 1;
        }

        return ret;
    }
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

    using typename GridBase<CELL, DIM>::BufferType;
    using GridBase<CELL, DIM>::topoDimensions;

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
        actualDimensions = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            actualDimensions[i] = box.dimensions[i];
        }
        actualDimensions += edgeRadii * 2;

        delegate.resize(
            actualDimensions.x(),
            actualDimensions.y(),
            actualDimensions.z());

        // init edges and interior
        delegate.callback(
            CUDASoAGridHelpers::SetContent<CELL, true>(
                delegate.get_data(),
                actualDimensions,
                edgeRadii,
                edgeCell,
                defaultCell));

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

    void setEdge(const CELL&)
    {
        // fixme
    }

    const CELL& getEdge() const
    {
        // fixme
    }

    CoordBox<DIM> boundingBox() const
    {
        // fixme
        return CoordBox<DIM>();
    }


    void saveRegion(BufferType *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        // fixme
    }

    void loadRegion(const BufferType& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        // fixme
    }

protected:
    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        // fixme
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        // fixme
    }

private:
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;

    static Coord<3> calcEdgeRadii()
    {
        return SoAGrid<CELL>::calcEdgeRadii();
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
