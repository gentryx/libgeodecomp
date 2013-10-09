#ifndef LIBGEODECOMP_IO_MOCKSTEERER_H
#define LIBGEODECOMP_IO_MOCKSTEERER_H

#include <sstream>
#include <libgeodecomp/io/steerer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MockSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef typename Steerer<CELL_TYPE>::SteererFeedback SteererFeedback;
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;
    static const int DIM = Topology::DIM;

    MockSteerer(const unsigned& period, std::stringstream *eventsBuffer)  :
        Steerer<CELL_TYPE>(period),
        eventsBuf(eventsBuffer)
    {
        (*eventsBuf) << "created, period = " << period << "\n";
    }

    virtual ~MockSteerer()
    {
        (*eventsBuf) << "deleted\n";
    }

    virtual void nextStep(
        GridType *grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        (*eventsBuf) << "nextStep(" << step << ", ";
        switch(event) {
        case STEERER_INITIALIZED:
            (*eventsBuf) << "STEERER_INITIALIZED";
            break;
        case STEERER_NEXT_STEP:
            (*eventsBuf) << "STEERER_NEXT_STEP";
            break;
        case STEERER_ALL_DONE:
            (*eventsBuf) << "STEERER_ALL_DONE";
            break;
        default:
            (*eventsBuf) << "unknown event";
            break;
        }

        (*eventsBuf) << ", " << rank << ", " << lastCall << ")\n";
    }

private:
    std::stringstream *eventsBuf;
};

}

#endif
