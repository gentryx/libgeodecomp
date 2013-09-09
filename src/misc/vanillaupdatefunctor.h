#ifndef LIBGEODECOMP_MISC_VANILLAUPDATEFUNCTOR_H
#define LIBGEODECOMP_MISC_VANILLAUPDATEFUNCTOR_H

namespace LibGeoDecomp {

/**
 * Updates a Streak of cells using the "vanilla" API (i.e.
 * LibGeoDecomp's classic cell interface which calls update() once per
 * cell and facilitates access to neighboring cells via a proxy object.
 */
template<typename CELL>
class VanillaUpdateFunctor
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    static const int DIM = Topology::DIM;

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        Coord<DIM> sourceCoord = streak.origin;
        Coord<DIM> targetCoord = targetOrigin;

        for (; sourceCoord.x() < streak.endX; ++sourceCoord.x()) {
            (*gridNew)[targetCoord].update(gridOld.getNeighborhood(sourceCoord), nanoStep);
            ++targetCoord.x();
        }
    }
};

}

#endif

