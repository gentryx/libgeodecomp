#ifndef LIBGEODECOMP_MISC_CONTAINERCELL_H
#define LIBGEODECOMP_MISC_CONTAINERCELL_H

#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/neighborhoodadapter.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

template<class CARGO, int SIZE, class TOPOLOGY=typename CARGO::Topology, typename KEY=int>
class ContainerCell {
public:
    friend class ContainerCellTest;

    typedef CARGO Cargo;
    typedef KEY Key;
    typedef TOPOLOGY Topology;
    typedef Cargo* Iterator;
    typedef Stencils::Moore<Topology::DIMENSIONS, 1> Stencil;

    class API : public CellAPITraits::Base
    {};


    inline static int nanoSteps()
    {
        return Cargo::nanoSteps();
    }

    const static int DIM = Topology::DIMENSIONS;
    const static int MAX_SIZE = SIZE;

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
            size--;
            return true;
        }

        return false;
    }

    inline Cargo *operator[](const Key& id) 
    {
        Key *end = ids + size;
        Key *pos = std::upper_bound(ids, end, id);        
        int offset = pos - ids;

        if (offset == 0)
            return 0;
        if (ids[offset - 1] == id)
            return cells + offset - 1;
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
        for (int i = 0; i < size; ++i) 
            cells[i].update(adapter, nanoStep);
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
        if (size == MAX_SIZE) 
            throw std::logic_error("ContainerCell capacity exeeded");
    }
};

}

#endif
