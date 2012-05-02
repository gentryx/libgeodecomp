#ifndef _libgeodecomp_misc_neighborhoodadapter_h_
#define _libgeodecomp_misc_neighborhoodadapter_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/neighborhoodadapter.h>

namespace LibGeoDecomp {

/**
 * This class is most useful for interfacing meshless codes with
 * LibGeoDecomp. It can retrieve cells matching a certain id from the
 * ContainerCells in the current neighborhood.
 */
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
        const Cargo *res = (*neighbors)[Coord<DIM>()][id];
            
        if (res)
            return *res;

        CoordBox<DIM> surroundingBox(Coord<DIM>::diagonal(-1), Coord<DIM>::diagonal(3));
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

}

#endif
