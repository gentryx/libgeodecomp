#ifndef _libgeodecomp_misc_containercell_h_
#define _libgeodecomp_misc_containercell_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>

namespace LibGeoDecomp {

template<class NEIGHBORHOOD, typename KEY, typename CARGO, int DIM>
class NeighborhoodAdapter
{
public:
    typedef KEY Key;
    typedef CARGO Cargo;

    NeighborhoodAdapter(NEIGHBORHOOD *_neighbors) :
        neighbors(_neighbors)
    {}

    const Cargo& operator[](const Key& id) 
    {
        Cargo *res = (*neighbors)[Coord<DIM>()][id];
            
        if (res)
            return *res;

        CoordBox<DIM> surroundingBox(CoordDiagonal<DIM>()(-1), CoordDiagonal<DIM>()(3));
        CoordBoxSequence<DIM> s = surroundingBox.sequence();
        while (s.hasNext()) {
            Coord<DIM> c = s.next();
            if (c != Coord<DIM>()) {
                res = (*neighbors)[c][id];
                if (res)
                    return *res;
            }
        }

        throw std::logic_error("id not found");
    }

    inline const Cargo& operator[](const Key& id) const
    {
        return (const_cast<NeighborhoodAdapter&>(*this))[id];
    }


private:
    NEIGHBORHOOD *neighbors;
};

template<class CARGO, int SIZE, class TOPOLOGY=typename CARGO::Topology, typename KEY=int>
class ContainerCell {
public:
    friend class ContainerCellTest;

    typedef CARGO Cargo;
    typedef KEY Key;
    typedef TOPOLOGY Topology;
    typedef Cargo* Iterator;

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

    inline Cargo *operator[](const Key& id) const
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
        NeighborhoodAdapter<NEIGHBORHOOD, Key, Cargo, DIM> adapter(&neighbors);;
        for (int i = 0; i < size; ++i) 
            cells[i].update(adapter, nanoStep);
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
