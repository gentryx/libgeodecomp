#ifndef LIBGEODECOMP_STORAGE_NEIGHBORHOODADAPTER_H
#define LIBGEODECOMP_STORAGE_NEIGHBORHOODADAPTER_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/collectioninterface.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

/**
 * This class is most useful for interfacing meshless codes with
 * LibGeoDecomp. It can retrieve cells matching a certain id from the
 * ContainerCells in the current neighborhood.
 */
template<
    class NEIGHBORHOOD,
    int DIM,
    typename COLLECTION_INTERFACE = CollectionInterface::PassThrough<typename NEIGHBORHOOD::Cell> >
class NeighborhoodAdapter
{
public:
    typedef typename NEIGHBORHOOD::Cell Cell;
    typedef typename COLLECTION_INTERFACE::Container::Key Key;
    typedef typename COLLECTION_INTERFACE::Container::Cargo Cargo;

    /**
     * The UNUSED parameter is here only for compatibility with N-body
     * codes based on BoxCell, which may be pulled in
     * MultiContainerCells, together with ContainerCells. BoxCells
     * need to spawn new particles and store those in the
     * writeContainer.
     */
    template<typename UNUSED>
    inline NeighborhoodAdapter(
        UNUSED *writeContainer,
        const NEIGHBORHOOD *neighbors) :
        neighbors(neighbors)
    {}

    /**
     * Will search neighboring containers for a Cargo which matches
     * the given ID.
     */
    const Cargo& operator[](const Key& id) const
    {
        const Cargo *res = COLLECTION_INTERFACE()((*neighbors)[Coord<DIM>()])[id];

        if (res) {
            return *res;
        }

        CoordBox<DIM> box(Coord<DIM>::diagonal(-1), Coord<DIM>::diagonal(3));
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if (*i != Coord<DIM>()) {
                res = COLLECTION_INTERFACE()((*neighbors)[*i])[id];
                if (res) {
                    return *res;
                }
            }
        }

        LOG(FAULT, "could not find id " << id << " in neighborhood");
        throw std::logic_error("id not found");
    }

private:
    const NEIGHBORHOOD *neighbors;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
