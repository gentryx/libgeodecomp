#ifndef LIBGEODECOMP_IO_MOCKSTEERER_H
#define LIBGEODECOMP_IO_MOCKSTEERER_H

#include <sstream>
#include <libgeodecomp/io/steerer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MockSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;

    MockSteerer(const unsigned& _period, std::stringstream *eventsBuffer)  :
        Steerer<CELL_TYPE>(_period),
        eventsBuf(eventsBuffer)
    { 
        (*eventsBuf) << "created, period = " << _period << "\n";
    }

    virtual ~MockSteerer() 
    { 
        (*eventsBuf) << "deleted\n"; 
    }

    virtual void nextStep(
        GridType *grid, 
        const Region<Topology::DIMENSIONS>& validRegion, 
        const unsigned& step) 
    {
        (*eventsBuf) << "nextStep(" << step << ")\n";
    }

private:
    std::stringstream *eventsBuf;
};

}

#endif
