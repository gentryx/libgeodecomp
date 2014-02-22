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
 * structure of the model.
 */
template<class CARGO, int SIZE, class TOPOLOGY=typename CARGO::Topology, typename KEY=int>
class ContainerCell
{
public:
    friend class ContainerCellTest;

    typedef CARGO Cargo;
    typedef KEY Key;
    typedef TOPOLOGY Topology;
    typedef Cargo *Iterator;

    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CARGO>::VALUE;
    const static int DIM = Topology::DIM;
    const static int MAX_SIZE = SIZE;

    class API :
        public APITraits::HasStencil<Stencils::Moore<Topology::DIM, 1> >,
        public APITraits::HasNanoSteps<NANO_STEPS>
    {};

    inline ContainerCell() :
        size(0)
    {}

    inline void insert(const Key& id, const Cargo& cell)
    {
        Key *end = ids + size;
        Key *pos = std::upper_bound(ids, end, id);

        if (pos == end) {
            checkSize();
            cells[size] = cell;
            ids[size++] = id;
            return;
        }

        int offset = pos - ids;
        if (offset > 0 && ids[offset - 1] == id) {
            cells[offset - 1] = cell;
            return;
        }

        checkSize();
        for (int i = size; i > offset; --i) {
            cells[i] = cells[i - 1];
            ids[i] = ids[i - 1];
        }

        cells[offset] = cell;
        ids[offset] = id;
        size++;
    }

    inline bool remove(const Key& id)
    {
        Cargo *pos = (*this)[id];
        if (pos) {
            int offset = pos - cells;
            for (int i = offset; i < size - 1; ++i) {
                cells[i] = cells[i + 1];
                ids[i] = ids[i + 1];
            }
            --size;
            return true;
        }

        return false;
    }

    inline Cargo *operator[](const Key& id)
    {
        Key *end = ids + size;
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

    inline Cargo *begin()
    {
        return cells;
    }

    inline Cargo *end()
    {
        return cells + size;
    }

    template<class NEIGHBORHOOD>
    inline void update(NEIGHBORHOOD neighbors, const int& nanoStep)
    {
        *this = neighbors[Coord<DIM>()];
        NeighborhoodAdapter<NEIGHBORHOOD, Key, Cargo, DIM> adapter(&neighbors);
        for (int i = 0; i < size; ++i) {
            cells[i].update(adapter, nanoStep);
        }
    }

    inline const Key *getIDs() const
    {
        return ids;
    }

    inline const int& getSize() const
    {
        return size;
    }

private:
    Key ids[SIZE];
    Cargo cells[SIZE];
    int size;

    inline void checkSize() const
    {
        if (size == MAX_SIZE) {
            throw std::logic_error("ContainerCell capacity exeeded");
        }
    }
};

}

#endif
