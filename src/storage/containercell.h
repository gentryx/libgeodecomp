#ifndef LIBGEODECOMP_STORAGE_CONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_CONTAINERCELL_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/storage/neighborhoodadapter.h>

namespace LibGeoDecomp {

/**
 * This class is useful for writing irregularly shaped codes with
 * LibGeoDecomp (e.g. meshfree or unstructured grids). It acts as an
 * adapter between the underlying, regular grid and the amorphous
 * structure of the model. Each entity of the model (of type CARGO)
 * needs to be assigned a unique KEY, which will be used for lookups.
 */
template<class CARGO, std::size_t SIZE, class TOPOLOGY=typename APITraits::SelectTopology<CARGO>::Value, typename KEY=int>
class ContainerCell
{
public:
    friend class ContainerCellTest;

    typedef CARGO Cargo;
    typedef KEY Key;
    typedef TOPOLOGY Topology;
    typedef Cargo *Iterator;
    typedef Cargo *iterator;
    typedef const Cargo *ConstIterator;
    typedef const Cargo *const_iterator;

    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CARGO>::VALUE;
    const static int DIM = Topology::DIM;
    const static std::size_t MAX_SIZE = SIZE;

    class API :
        public APITraits::HasStencil<Stencils::Moore<Topology::DIM, 1> >,
        public APITraits::HasNanoSteps<NANO_STEPS>
    {};

    inline ContainerCell() :
        numElements(0)
    {}

    inline void insert(const Key& id, const Cargo& cell)
    {
        Key *end = ids + numElements;
        Key *pos = std::upper_bound(ids, end, id);

        if (pos == end) {
            checkSize();
            cells[numElements] = cell;
            ids[numElements++] = id;
            return;
        }

        int offset = pos - ids;
        if (offset > 0 && ids[offset - 1] == id) {
            cells[offset - 1] = cell;
            return;
        }

        checkSize();
        for (int i = numElements; i > offset; --i) {
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
            int offset = pos - cells;
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

    template<class NEIGHBORHOOD>
    inline void update(NEIGHBORHOOD neighbors, const int& nanoStep)
    {
        *this = neighbors[Coord<DIM>()];
        NeighborhoodAdapter<NEIGHBORHOOD, Key, Cargo, DIM> adapter(&neighbors);

        for (std::size_t i = 0; i < numElements; ++i) {
            cells[i].update(adapter, nanoStep);
        }
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

}

#endif
