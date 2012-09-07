#ifndef _libgeodecomp_io_mocksteerer_h_
#define _libgeodecomp_io_mocksteerer_h_

#include <sstream>
#include <libgeodecomp/io/steerer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MockSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;

    MockSteerer(const unsigned& _period, std::ostringstream *eventsBuffer)  :
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
    std::ostringstream *eventsBuf;
};

}

#endif
