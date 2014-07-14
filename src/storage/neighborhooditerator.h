#ifndef LIBGEODECOMP_STORAGE_NEIGHBORHOODITERATOR_H
#define LIBGEODECOMP_STORAGE_NEIGHBORHOODITERATOR_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>

namespace LibGeoDecomp {

/**
 * This class is meant to be used with BoxCell and alike to interface
 * MD and n-body codes with our standard neighborhood types. It allows
 * models to transparently traverse all particles in their neighboring
 * containers.
 */
template<class NEIGHBORHOOD, typename CARGO, int DIM>
class NeighborhoodIterator
{
public:
    friend class NeighborhoodIteratorTest;

    typedef NEIGHBORHOOD Neighborhood;
    typedef typename Neighborhood::Cell Cell;
    typedef typename Cell::const_iterator CellIterator;
    typedef typename Cell::value_type Particle;

    inline NeighborhoodIterator(
        const Neighborhood& hood,
        const Coord<DIM>& coord,
        const CellIterator& iterator) :
        hood(hood),
        boxIterator(
            typename CoordBox<DIM>::Iterator(
                Coord<DIM>::diagonal(-1),
                coord,
                Coord<DIM>::diagonal(3))),
        end(
            CoordBox<DIM>(
                Coord<DIM>::diagonal(-1),
                Coord<DIM>::diagonal(3)).end()),
        cell(&hood[coord]),
        iterator(iterator)
    {}

    const Particle& operator*() const
    {
        return *iterator;
    }

    inline void operator++()
    {
        ++iterator;

        if (iterator == cell->end()) {
            ++boxIterator;

            // this check is required to avoid dereferentiation of the
            // neighborhood with an out-of-range coordinate.
            if (boxIterator != end) {
                cell = &hood[*boxIterator];
                iterator = cell->begin();
            }
        }
    }

    inline bool operator==(const NeighborhoodIterator& other) const
    {
        return (cell == other.cell) && (iterator == other.iterator);
    }


    inline bool operator!=(const NeighborhoodIterator& other) const
    {
        return !(*this == other);
    }

private:
    const Neighborhood& hood;
    typename CoordBox<DIM>::Iterator boxIterator;
    typename CoordBox<DIM>::Iterator end;
    const Cell *cell;
    CellIterator iterator;

};

}

#endif
