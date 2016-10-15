#ifndef LIBGEODECOMP_STORAGE_GRIDBASE_H
#define LIBGEODECOMP_STORAGE_GRIDBASE_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/memorylocation.h>
#include <libgeodecomp/storage/selector.h>

namespace LibGeoDecomp {

namespace GridBaseHelpers {

/**
 * We cannot define the functions below inside GridBase as they would
 * clash with the variants that use the cell type in their signature
 * if CELL == char. We could disallow char as a template parameter to
 * GridBase and friends, but that seems unnatural.
 */
template<int DIM>
class LoadSaveRegionCharInterface
{
public:
    virtual ~LoadSaveRegionCharInterface()
    {}

    /**
     * This will typically be implemented by grids with Struct of
     * Arrays (SoA) layout.
     */
    virtual void saveRegion(std::vector<char> *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        throw std::logic_error("saveRegion not implemented for char buffers, not an SoA grid?");
    }

    /**
     * This will typically be implemented by grids with Struct of
     * Arrays (SoA) layout.
     */
    virtual void loadRegion(const std::vector<char>& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        throw std::logic_error("loadRegion not implemented for char buffers, not an SoA grid?");
    }
};

}

template<typename CELL, int DIM, typename WEIGHT_TYPE>
class ProxyGrid;

/**
 * This is an abstract base class for all grid classes. It's generic
 * because all methods are virtual, but single element access is not
 * very efficient -- for the same reason. Fast bulk access to members
 * of the cells is granted by saveMember()/loadMember().
 *
 * CELLTYPE is the type of the grid's elements. DIMENSIONS is the
 * dimensionality of the regular grid, not the extent. Unstructured
 * grids have a DIMENSIONALITY of 1. WEIGHT_TYPE is relevant only for
 * unstructured grids and determines the type of the edge weights
 * stored with the adjacency.
 */
template<typename CELL, int DIMENSIONS, typename WEIGHT_TYPE = double>
class GridBase : GridBaseHelpers::LoadSaveRegionCharInterface<DIMENSIONS>
{
public:
    friend class ProxyGrid<CELL, DIMENSIONS, WEIGHT_TYPE>;
    typedef CELL CellType;

    using GridBaseHelpers::LoadSaveRegionCharInterface<DIMENSIONS>::saveRegion;
    using GridBaseHelpers::LoadSaveRegionCharInterface<DIMENSIONS>::loadRegion;

    const static int DIM = DIMENSIONS;

    explicit inline GridBase(const Coord<DIM>& topoDimensions = Coord<DIM>()) :
        topoDimensions(topoDimensions)
    {}

    virtual ~GridBase()
    {}

    /**
     * Changes the dimension and offset of the grid.
     */
    virtual void resize(const CoordBox<DIM>&) = 0;

    /**
     * Copies a single cell into the grid at the given coordinate
     */
    virtual void set(const Coord<DIM>&, const CELL&) = 0;

    /**
     * Copies a row of cells into the grid. The pointer is expected to
     * point to a memory location with at least as many cells as the
     * streak specifies.
     */
    virtual void set(const Streak<DIM>&, const CELL*) = 0;

    /**
     * Copies out a single cell from the grid.
     */
    virtual CELL get(const Coord<DIM>&) const = 0;

    /**
     * Copies as many cells as given by the Streak, starting at its
     * origin, to the pointer. The target is expected to point to a
     * memory location with sufficient space.
     */
    virtual void get(const Streak<DIM>&, CELL *) const = 0;

    /**
     * The edge cell is returned for out-of-bounds accesses on
     * non-periodic boundaries. If can be set via this function.
     */
    virtual void setEdge(const CELL&) = 0;

    /**
     * Reading counterpart for setEdge().
     */
    virtual const CELL& getEdge() const = 0;

    /**
     * Returns the extent of the grid (origin and dimension).
     */
    virtual CoordBox<DIM> boundingBox() const = 0;

    /**
     * Returns the set of coordinates contained by the grid. For
     * regular grids this can be expected to be identical with the
     * bounding box. Unstructured grids may opt to store less cells in
     * order to increase space efficiency.
     */
    virtual const Region<DIM>& boundingRegion()
    {
        myBoundingRegion.clear();
        myBoundingRegion << boundingBox();
        return myBoundingRegion;
    }

    /**
     * Extract cells specified by the Region and serialize them in the
     * given buffer. An optional offset will be added to all
     * coordinates in the Region.
     *
     * This function is typically implemented by Array of Structs
     * (AoS) grids. SoA grids implement the variant that uses char
     * buffers.
     */
    virtual void saveRegion(std::vector<CELL> *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        throw std::logic_error("loadRegion not implemented for buffers of type CELL, not an AoS grid?");
    }

    /**
     * Load cells from the buffer and store them at the coordinates
     * specified in region. The Region may be translated by an
     * optional offset.
     *
     * This function is typically implemented by Array of Structs
     * (AoS) grids. SoA grids implement the variant that uses char
     * buffers.
     */
    virtual void loadRegion(const std::vector<CELL>& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        throw std::logic_error("loadRegion not implemented for buffers of type CELL, not an AoS grid?");
    }

    Coord<DIM> dimensions() const
    {
        return boundingBox().dimensions;
    }

    bool operator==(const GridBase<CELL, DIMENSIONS>& other) const
    {
        if (getEdge() != other.getEdge()) {
            return false;
        }

        CoordBox<DIM> box = boundingBox();
        if (box != other.boundingBox()) {
            return false;
        }

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if (get(*i) != other.get(*i)) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const GridBase<CELL, DIMENSIONS>& other) const
    {
        return !(*this == other);
    }

    /**
     * Allows the user to extract a single member variable of all
     * cells within the given region. Assumes that target points to an area with sufficient space.
     */
    template<typename MEMBER>
    void saveMember(
        MEMBER *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        if (!selector.template checkTypeID<MEMBER>()) {
            throw std::invalid_argument("cannot save member as selector was created for different type");
        }

        saveMemberImplementation(reinterpret_cast<char*>(target), targetLocation, selector, region);
    }

    /**
     * Same as saveMember(), but sans the type checking. Useful in
     * Writers and other components that might not know about the
     * variable's type.
     */
    void saveMemberUnchecked(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        saveMemberImplementation(target, targetLocation, selector, region);
    }

    /**
     * Used for bulk-setting of single member variables. Assumes that
     * source contains as many instances of the member as region
     * contains coordinates.
     */
    template<typename MEMBER>
    void loadMember(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        if (!selector.template checkTypeID<MEMBER>()) {
            throw std::invalid_argument("cannot load member as selector was created for different type");
        }

        loadMemberImplementation(reinterpret_cast<const char*>(source), sourceLocation, selector, region);
    }

    /**
     * Through this function the weights of the edges on unstructured
     * grids can be set. Unavailable on regular grids.
     *
     * fixme: type of matrix is terrible
     */
    virtual void setWeights(std::size_t matrixID, const std::map<Coord<2>, WEIGHT_TYPE>& matrix)
    {
        throw std::logic_error("edge weights cannot be set on this grid type");
    }

    const Coord<DIM>& topologicalDimensions() const
    {
        return topoDimensions;
    }

    /**
     * This function can be used to obtain a Region which implements
     * transformations required by the grid for efficient element
     * access. Regular grids, e.g. DisplacedGrid, renerally only
     * require an affine transformation which can be done efficiently
     * in place (hence no remapping is required), but unstructured
     * grids may need extensive reordering due to sparse IDs and the
     * SELL-C-Sigma format.
     */
    virtual
    Region<DIM> remapRegion(const Region<DIM>& region) const
    {
        return region;
    }

protected:
    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const = 0;

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) = 0;

    Coord<DIM> topoDimensions;
    Region<DIM> myBoundingRegion;
};

}

#endif
