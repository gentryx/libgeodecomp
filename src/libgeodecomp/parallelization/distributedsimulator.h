#ifndef LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_DISTRIBUTEDSIMULATOR_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/sharedptr.h>
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
    typedef typename SharedPtr<ParallelWriter<CELL_TYPE> >::Type WriterPtr;
    typedef std::vector<WriterPtr> WriterVector;
    typedef typename Simulator<CELL_TYPE>::InitPtr InitPtr;

    using Simulator<CELL_TYPE>::chronometer;

    inline explicit DistributedSimulator(Initializer<CELL_TYPE> *initializer) :
        Simulator<CELL_TYPE>(initializer)
    {}

    inline explicit DistributedSimulator(InitPtr initializer) :
        Simulator<CELL_TYPE>(initializer)
    {}

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

    /**
     * register  writer which will observe the simulation. The
     * DistributedSimulator will assume that it now owns the
     * ParallelWriter, so it'll delete it upon destruction.
     */
    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        writers << WriterPtr(writer);
    }

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

protected:
    WriterVector writers;

};

}

#endif
