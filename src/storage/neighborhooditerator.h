#ifndef LIBGEODECOMP_STORAGE_NEIGHBORHOODITERATOR_H
#define LIBGEODECOMP_STORAGE_NEIGHBORHOODITERATOR_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/storage/collectioninterface.h>

namespace LibGeoDecomp {

namespace NeighborhoodIteratorHelpers {

/**
 * This is a shim to let MultiContainerCell handle members of type
 * BoxCell (which use this NeighborhoodIterator for accessing the
 * surrounding particles) identically to ContainerCells (which
 * rely on NeighborhoodAdapter)
 */
template<typename CONTAINER, typename NEIGHBORHOOD_ITERATOR>
class Adapter
{
 public:
    typedef NEIGHBORHOOD_ITERATOR Iterator;

    inline
    explicit Adapter(
        CONTAINER *container,
        const typename Iterator::Neighborhood *hood) :
        container(container),
        myBegin(Iterator::begin(container, *hood)),
        myEnd(Iterator::end(container, *hood))
    {}

    inline
    const Iterator& begin() const
    {
        return myBegin;
    }

    inline
    const Iterator& end() const
    {
        return myEnd;
    }

 private:
    CONTAINER *container;
    Iterator myBegin;
    Iterator myEnd;
};


}

/**
 * This class is meant to be used with BoxCell and alike to interface
 * MD and n-body codes with our standard neighborhood types. It allows
 * models to transparently traverse all particles in their neighboring
 * containers.
 */
template<
    typename WRITE_CONTAINER,
    typename NEIGHBORHOOD,
    int DIM,
    typename COLLECTION_INTERFACE=CollectionInterface::PassThrough<typename NEIGHBORHOOD::Cell> >
class NeighborhoodIterator
{
public:
    friend class NeighborhoodIteratorTest;

    typedef NEIGHBORHOOD Neighborhood;
    typedef typename Neighborhood::Cell Cell;
    typedef typename COLLECTION_INTERFACE::Container Container;
    typedef typename COLLECTION_INTERFACE::Container::const_iterator CellIterator;
    typedef typename COLLECTION_INTERFACE::Container::value_type Particle;

    inline NeighborhoodIterator(
        WRITE_CONTAINER *writeContainer,
        const Neighborhood& hood,
        const Coord<DIM>& coord,
        const CellIterator& iterator) :
        writeContainer(writeContainer),
        hood(hood),
        boxIterator(
            typename CoordBox<DIM>::Iterator(
                Coord<DIM>::diagonal(-1),
                coord,
                Coord<DIM>::diagonal(3))),
        endIterator(
            CoordBox<DIM>(
                Coord<DIM>::diagonal(-1),
                Coord<DIM>::diagonal(3)).end()),
        cell(&hood[coord]),
        iterator(iterator)
    {}

    static inline NeighborhoodIterator begin(
        WRITE_CONTAINER *writeContainer,
        const Neighborhood& hood)
    {
        CoordBox<DIM> box(Coord<DIM>::diagonal(-1), Coord<DIM>::diagonal(3));

        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {
            if (COLLECTION_INTERFACE()(hood[*i]).size() > 0) {
                return NeighborhoodIterator(
                    writeContainer,
                    hood,
                    *i,
                    COLLECTION_INTERFACE()(hood[*i]).begin());
            }
        }

        Coord<DIM> endCoord = Coord<DIM>::diagonal(1);
        return NeighborhoodIterator(
            writeContainer,
            hood,
            endCoord,
            COLLECTION_INTERFACE()(hood[endCoord]).end());
    }

    static inline NeighborhoodIterator end(
        WRITE_CONTAINER *writeContainer,
        const Neighborhood& hood)
    {
        return NeighborhoodIterator(
            writeContainer,
            hood,
            Coord<DIM>::diagonal(1),
            COLLECTION_INTERFACE()(hood[Coord<DIM>::diagonal(1)]).end());
    }

    inline const Particle& operator*() const
    {
        return *iterator;
    }

    inline CellIterator operator->() const
    {
        return iterator;
    }

    inline void operator++()
    {
        ++iterator;

        while (iterator == COLLECTION_INTERFACE()(*cell).end()) {
            ++boxIterator;

            // this check is required to avoid dereferentiation of the
            // neighborhood with an out-of-range coordinate.
            if (boxIterator == endIterator) {
                return;
            }

            cell = &hood[*boxIterator];
            iterator = COLLECTION_INTERFACE()(*cell).begin();
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
    WRITE_CONTAINER *writeContainer;
    const Neighborhood& hood;
    typename CoordBox<DIM>::Iterator boxIterator;
    typename CoordBox<DIM>::Iterator endIterator;
    const Cell *cell;
    CellIterator iterator;

};

}

#endif
