#ifndef LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/simulator.h>

namespace LibGeoDecomp {

template<class CELL_TYPE> class ParallelWriter;

// fixme: add short doxygen doc for every class
template<typename CELL_TYPE>
class DistributedSimulator : public Simulator<CELL_TYPE>
{
public:
    typedef typename Simulator<CELL_TYPE>::Topology Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
    typedef SuperVector<boost::shared_ptr<ParallelWriter<CELL_TYPE> > > WriterVector;

    inline DistributedSimulator(Initializer<CELL_TYPE> *initializer) :
        Simulator<CELL_TYPE>(initializer)
    {}

    /**
     * register  writer which will observe the simulation. The
     * DistributedSimulator will assume that it now owns the
     * ParallelWriter, so it'll delete it upon destruction.
     */
    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        writers << boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer);
    }

protected:
    WriterVector writers;

};

}

#endif
