#ifndef _libgeodecomp_misc_vanillaupdatefunctor_h_
#define _libgeodecomp_misc_vanillaupdatefunctor_h_

namespace LibGeoDecomp {

/**
 * updates a Streak of cells using the "vanilla" API (i.e.
 * LibGeoDecomp's classic cell interface which calls update() once per
 * cell and facilitates access to neighboring cells via a proxy object.
 */
template<typename CELL>
class VanillaUpdateFunctor
{
public:
    static const int DIM = CELL::Topology::DIMENSIONS;

    template<typename GRID>
    void operator()(
        const Streak<DIM>& streak,
        const GRID& gridOld,
        GRID *gridNew,
        unsigned nanoStep) 
    {
        Coord<DIM> c = streak.origin;
        for (; c.x() < streak.endX; ++c.x()) {
            (*gridNew)[c].update(gridOld.getNeighborhood(c), nanoStep);
        }
    }
};

}

#endif

