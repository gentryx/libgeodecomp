#ifndef LIBGEODECOMP_IO_STEERER_H
#define LIBGEODECOMP_IO_STEERER_H

#include <libgeodecomp/misc/gridbase.h>
#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class Steerer
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;

    Steerer(const unsigned _period) :
        period(_period)
    {}

    virtual ~Steerer() 
    {}

    /**
     * is a callback which gives the Steerer access to a Simulator's
     * grid. The part which is accessible via \p grid is specified in
     * \p validRegion. The current time step is given in \p step. This
     * function may be called multiple times per step (e.g. seperately
     * for inner ghost zones and inner set (which is equivalent to the
     * interface of ParallelWriter).
     */
    virtual void nextStep(
        GridType *grid, 
        const Region<Topology::DIM>& validRegion, 
        const unsigned& step) =0;

    const unsigned& getPeriod() const
    {
        return period;
    }

protected:
    unsigned period;
};

}

#endif
