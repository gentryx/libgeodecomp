#ifndef LIBGEODECOMP_PARALLELIZATION_MONOLITHICSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_MONOLITHICSIMULATOR_H

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/simulator.h>

namespace LibGeoDecomp {

template<class CELL_TYPE> class Writer;

// Padding is fine
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * A MonolithicSimulator is a Simulator which runs on a
 */
template<typename CELL_TYPE>
class MonolithicSimulator : public Simulator<CELL_TYPE>
{
public:
    using typename Simulator<CELL_TYPE>::Topology;
    using Simulator<CELL_TYPE>::chronometer;
    typedef typename SharedPtr<Writer<CELL_TYPE> >::Type WriterPtr;
    typedef std::vector<WriterPtr> WriterVector;

    inline explicit MonolithicSimulator(Initializer<CELL_TYPE> *initializer) :
        Simulator<CELL_TYPE>(initializer)
    {}

    /**
     * Returns the current grid.
     */
    virtual const typename Simulator<CELL_TYPE>::GridType *getGrid() = 0;

    /**
     * register  writer which will observe the simulation. The
     * MonolithicSimulator will assume that it now owns the Writer, so
     * it'll delete it upon destruction.
     */
    virtual void addWriter(Writer<CELL_TYPE> *writer)
    {
        writers.push_back(WriterPtr(writer));
    }

    std::vector<Chronometer> gatherStatistics()
    {
        return std::vector<Chronometer>(1, chronometer);
    }

protected:
    WriterVector writers;

};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
