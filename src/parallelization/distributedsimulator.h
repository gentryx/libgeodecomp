#ifndef LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/parallelization/simulator.h>
#include <libgeodecomp/storage/displacedgrid.h>

namespace LibGeoDecomp {

template<class CELL_TYPE> class ParallelWriter;

/**
 * This class encompasses all Simulators which can run on multiple
 * nodes (e.g. using MPI or HPX for synchronization).
 */
template<typename CELL_TYPE>
class DistributedSimulator : public Simulator<CELL_TYPE>
{
public:
    typedef typename Simulator<CELL_TYPE>::Topology Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
    typedef std::vector<boost::shared_ptr<ParallelWriter<CELL_TYPE> > > WriterVector;
    using Simulator<CELL_TYPE>::chronometer;

    inline explicit DistributedSimulator(Initializer<CELL_TYPE> *initializer) :
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
