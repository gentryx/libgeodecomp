#ifndef LIBGEODECOMP_PARALLELIZATION_MONOLITHICSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_MONOLITHICSIMULATOR_H

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/simulator.h>

namespace LibGeoDecomp {

template<class CELL_TYPE> class Writer;

template<typename CELL_TYPE>
class MonolithicSimulator : public Simulator<CELL_TYPE>
{
public:
    typedef std::vector<boost::shared_ptr<Writer<CELL_TYPE> > > WriterVector;

    inline MonolithicSimulator(Initializer<CELL_TYPE> *_initializer) :
        Simulator<CELL_TYPE>(_initializer)
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
        writers.push_back(boost::shared_ptr<Writer<CELL_TYPE> >(writer));
    }

protected:
    WriterVector writers;

};

}

#endif
