#ifndef LIBGEODECOMP_STORAGE_CONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_CONTAINERCELL_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/storage/neighborhoodadapter.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * This class is useful for writing irregularly shaped codes with
 * LibGeoDecomp (e.g. meshfree or unstructured grids). It acts as an
 * adapter between the underlying, regular grid and the amorphous
 * structure of the model. Each entity of the model (of type CARGO)
 * needs to be assigned a unique KEY, which will be used for lookups.
 *
 * If your model doesn't access neighboring cells via IDs but rather
 * all neighbors within a certain radius, then BoxCell is a better
 * choice.
 */
template<typename CARGO, std::size_t SIZE, typename KEY = int>
class ContainerCell
{
public:
    friend class ContainerCellTest;

    typedef CARGO Cargo;
    typedef CARGO value_type;
    typedef KEY Key;
    typedef typename APITraits::SelectTopology<CARGO>::Value Topology;
    typedef Cargo *Iterator;
    typedef Cargo *iterator;
    typedef const Cargo *ConstIterator;
    typedef const Cargo *const_iterator;

    const static int DIM = Topology::DIM;
    const static std::size_t MAX_SIZE = SIZE;

    template<
        // currently unused as we don't yet allow unstructed grid
        // codes to add cells to ContainerCells at runtime:
        typename WRITE_CONTAINER,
        typename NEIGHBORHOOD,
        typename COLLECTION_INTERFACE>
    class NeighborhoodAdapter
    {
    public:
        typedef typename LibGeoDecomp::NeighborhoodAdapter<NEIGHBORHOOD, DIM, COLLECTION_INTERFACE> Value;
    };

    class API :
        public APITraits::SelectAPI<CARGO>::Value,
        public APITraits::HasStencil<Stencils::Moore<Topology::DIM, 1> >
    {};

    inline ContainerCell() :
        numElements(0)
    {}

    inline void insert(const Key& id, const Cargo& cell)
    {
        Key *end = ids + numElements;
        Key *pos = std::upper_bound(ids, end, id);

        std::size_t offset = static_cast<std::size_t>(pos - ids);
        if ((offset > 0) && (ids[offset - 1] == id)) {
            cells[offset - 1] = cell;
            return;
        }

        checkSize();

        if (pos == end) {
            cells[numElements] = cell;
            ids[numElements++] = id;
            return;
        }

        for (std::size_t i = numElements; i > offset; --i) {
            cells[i] = cells[i - 1];
            ids[i] = ids[i - 1];
        }

        cells[offset] = cell;
        ids[offset] = id;
        ++numElements;
    }

    inline bool remove(const Key& id)
    {
        Cargo *pos = (*this)[id];
        if (pos) {
            std::size_t offset = static_cast<std::size_t>(pos - cells);
            for (std::size_t i = offset; i < numElements - 1; ++i) {
                cells[i] = cells[i + 1];
                ids[i] = ids[i + 1];
            }
            --numElements;
            return true;
        }

        return false;
    }

    inline Cargo *operator[](const Key& id)
    {
        Key *end = ids + numElements;
        Key *pos = std::upper_bound(ids, end, id);
        int offset = pos - ids;

        if (offset == 0) {
            return 0;
        }

        if (ids[offset - 1] == id) {
            return cells + offset - 1;
        }

        return 0;
    }

    inline const Cargo *operator[](const Key& id) const
    {
        return (const_cast<ContainerCell&>(*this))[id];
    }

    inline void clear()
    {
        numElements = 0;
    }

    inline Cargo *begin()
    {
        return cells;
    }

    inline const Cargo *begin() const
    {
        return cells;
    }

    inline Cargo *end()
    {
        return cells + numElements;
    }

    inline const Cargo *end() const
    {
        return cells + numElements;
    }

    inline std::size_t size() const
    {
        return numElements;
    }

    /**
     * The normal update() will copy its state from last time step so
     * all cargo items are well initialized. Otherwise the current
     * container wouldn't even know the IDs of the items which need
     * updating.
     */
    template<class HOOD>
    inline void update(const HOOD& neighbors, const unsigned nanoStep)
    {
        typedef CollectionInterface::PassThrough<typename HOOD::Cell> PassThroughType;
        typedef typename NeighborhoodAdapter<ContainerCell, HOOD, PassThroughType>::Value NeighborhoodAdapterType;
        NeighborhoodAdapterType adapter(this, &neighbors);

        copyOver(neighbors[Coord<DIM>()], adapter, nanoStep);
        updateCargo(adapter, nanoStep);
    }

    template<class HOOD_SELF>
    inline void copyOver(const ContainerCell& oldSelf, HOOD_SELF& ownNeighbors, const unsigned nanoStep)
    {
        *this = oldSelf;
    }

    /**
     * Assuming that some external entity has already taken care of
     * initializing this container's cargo, we also provide
     * updateCargo(), which doesn't copy over the old state:
     */
    template<class HOOD_ALL>
    inline void updateCargo(HOOD_ALL& allNeighbors, const unsigned nanoStep)
    {
        for (std::size_t i = 0; i < numElements; ++i) {
            cells[i].update(allNeighbors, nanoStep);
        }
    }

    template<class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & ids & cells & numElements;
    }

    inline const Key *getIDs() const
    {
        return ids;
    }

private:

    Key ids[SIZE];
    Cargo cells[SIZE];
    std::size_t numElements;

    inline void checkSize() const
    {
        if (numElements == MAX_SIZE) {
            throw std::logic_error("ContainerCell capacity exeeded");
        }
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

template<typename ARCHIVE, typename CARGO, std::size_t SIZE, typename KEY>
void serialize(ARCHIVE& ar, ContainerCell<CARGO, SIZE, KEY> cargoCell, unsigned v)
{
    cargoCell.serialize(ar, v);
}

}

#endif
