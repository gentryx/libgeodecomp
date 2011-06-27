#ifndef _libgeodecomp_parallelization_monolithicsimulator_h_
#define _libgeodecomp_parallelization_monolithicsimulator_h_

#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/simulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class Writer; 

template<typename CELL_TYPE>
class Simulator;

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
     * register @a writer to receive the Writer callbacks.
     */
    virtual void registerWriter(Writer<CELL_TYPE> *writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL_TYPE> >(writer));
    }

    /**
     * @return currently registered Writers.
     */
    virtual WriterVector getWriters() const
    {
        return writers;
    }

protected:
    WriterVector writers;

};

}

#endif
