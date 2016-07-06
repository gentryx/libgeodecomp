#ifndef LIBGEODECOMP_STORAGE_CUDAGRID_H
#define LIBGEODECOMP_STORAGE_CUDAGRID_H

#include <libflatarray/cuda_array.hpp>

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/gridbase.h>

#include <cuda.h>

namespace LibGeoDecomp {

/**
 * A lightweight AoS-style (Array of Structs) grid class, to manage
 * memory allocation and bulk data transfer on NVDIA GPUs.
 */
template<typename CELL_TYPE,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDAGrid : public GridBase<CELL_TYPE, TOPOLOGY::DIM>
{
public:
    friend class CUDAGridTest;

    typedef CELL_TYPE CellType;
    typedef TOPOLOGY Topology;

    static const int DIM = Topology::DIM;
    using GridBase<CellType, DIM>::topoDimensions;
    using GridBase<CellType, DIM>::saveRegion;
    using GridBase<CellType, DIM>::loadRegion;

    explicit inline CUDAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CellType& defaultCell = CellType(),
        const CellType& edgeCell = CellType(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CellType, DIM>(topologicalDimensions),
        box(box),
        array(box.dimensions.prod(), defaultCell),
        edgeCellStore(1, edgeCell),
        hostEdgeCell(edgeCell)
    {}

    inline
    void resize(const CoordBox<DIM>& newBox)
    {
        box = newBox;
        array.resize(box.dimensions.prod());
    }

    inline
    void set(const Coord<DIM>& coord, const CELL_TYPE& source)
    {
        std::size_t length = sizeof(CellType);
        CELL_TYPE *target = address(coord);
        if (target == edgeCellStore.data()) {
            hostEdgeCell = source;
            edgeCellStore.load(&source);
        }
        cudaMemcpy(target, &source, length, cudaMemcpyHostToDevice);
    }

    inline
    void set(const Streak<DIM>& streak, const CELL_TYPE *source)
    {
        std::size_t length = streak.length() * sizeof(CellType);
        cudaMemcpy(address(streak.origin), source, length, cudaMemcpyHostToDevice);
    }

    inline
    CELL_TYPE get(const Coord<DIM>& coord) const
    {
        CELL_TYPE ret;
        std::size_t length = sizeof(CellType);
        cudaMemcpy(&ret, const_cast<CellType*>(address(coord)), length, cudaMemcpyDeviceToHost);
        return ret;
    }

    inline
    void get(const Streak<DIM>& streak, CELL_TYPE *target) const
    {
        std::size_t length = streak.length() * sizeof(CellType);
        cudaMemcpy(target, const_cast<CellType*>(address(streak.origin)), length, cudaMemcpyDeviceToHost);
    }

    void setEdge(const CELL_TYPE& cell)
    {
        hostEdgeCell = cell;
        edgeCellStore.load(&cell);
    }

    const CELL_TYPE& getEdge() const
    {
        return hostEdgeCell;
    }

    CoordBox<DIM> boundingBox() const
    {
        return box;
    }

    void saveRegion(std::vector<CELL_TYPE> *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        // fixme: check size of buffer
        CellType *cursor = buffer->data();

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(cursor, const_cast<CellType*>(address(i->origin + offset)), length, cudaMemcpyDeviceToHost);
            cursor += i->length();
        }
    }

    void loadRegion(const std::vector<CELL_TYPE>& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        // fixme: check size of buffer
        const CellType *cursor = buffer.data();

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(address(i->origin + offset), cursor, length, cudaMemcpyHostToDevice);
            cursor += i->length();
        }
    }

    template<typename GRID_TYPE, typename REGION>
    void saveRegion(GRID_TYPE *target, const REGION& region) const
    {
        for (typename REGION::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(&(*target)[i->origin], const_cast<CellType*>(address(i->origin)), length, cudaMemcpyDeviceToHost);
        }
    }

    template<typename GRID_TYPE, typename REGION>
    void loadRegion(const GRID_TYPE& source, const REGION& region)
    {
        for (typename REGION::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(address(i->origin), &source[i->origin], length, cudaMemcpyHostToDevice);
        }
    }

    __host__ __device__
    const CoordBox<DIM>& boundingBox()
    {
        return box;
    }

    __host__ __device__
    CellType *data()
    {
        return array.data();
    }

    __host__ __device__
    const CellType *data() const
    {
        return array.data();
    }

    __host__ __device__
    CellType *edgeCell()
    {
        return edgeCellStore.data();
    }

    __host__ __device__
    const CellType *edgeCell() const
    {
        return edgeCellStore.data();
    }

protected:
    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL_TYPE>& selector,
        const Region<DIM>& region) const
    {
        char *targetCursor = target;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberOut(
                address(i->origin),
                MemoryLocation::CUDA_DEVICE,
                targetCursor,
                targetLocation,
                i->length());

            targetCursor += i->length() * selector.sizeOfExternal();
        }
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL_TYPE>& selector,
        const Region<DIM>& region)
    {
        const char *sourceCursor = source;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberIn(
                sourceCursor,
                sourceLocation,
                address(i->origin) ,
                MemoryLocation::CUDA_DEVICE,
                i->length());

            sourceCursor += i->length() * selector.sizeOfExternal();
        }
    }

private:
    CoordBox<DIM> box;
    LibFlatArray::cuda_array<CellType> array;
    LibFlatArray::cuda_array<CellType> edgeCellStore;
    CellType hostEdgeCell;

    std::size_t byteSize() const
    {
        return box.dimensions.prod() * sizeof(CellType);
    }

    Coord<DIM> toRelativeCoord(const Coord<DIM>& absoluteCoord) const
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        return relativeCoord;
    }

    CellType *address(const Coord<DIM>& absoluteCoord)
    {
        Coord<DIM> relativeCoord = toRelativeCoord(absoluteCoord);

        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            return edgeCellStore.data();
        }

        return data() + relativeCoord.toIndex(box.dimensions);
    }

    const CellType *address(const Coord<DIM>& absoluteCoord) const
    {
        Coord<DIM> relativeCoord = toRelativeCoord(absoluteCoord);

        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            return edgeCellStore.data();
        }

        return data() + relativeCoord.toIndex(box.dimensions);
    }
};

}

#endif
