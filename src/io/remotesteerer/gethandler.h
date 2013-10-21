#ifndef LIBGEODECOMP_IO_REMOTESTEERER_GETHANDLER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_GETHANDLER_H

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/io/remotesteerer/handler.h>
#include <libgeodecomp/storage/dataaccessor.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

template<typename CELL_TYPE, typename MEMBER_TYPE>
class GetHandler : public Handler<CELL_TYPE>
{
public:
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
    typedef boost::shared_ptr<DataAccessor<CELL_TYPE, MEMBER_TYPE> > AccessorPtr;

    static const int DIM = Topology::DIM;

    GetHandler(AccessorPtr accessor) :
        Handler<CELL_TYPE>("get_" + accessor->name()),
        accessor(accessor)
    {}

    virtual bool operator()(const StringVec& parameters, Pipe& pipe, GridType *grid, const Region<DIM>& validRegion, unsigned step)
    {
        LOG(DBG, "GetHander::operator()(" << parameters << " step: " << step << ")");

        Coord<DIM> c;
        int index;

        unsigned time = StringOps::atoi(parameters[0]);

        for (index = 1; index < DIM; ++index) {
            c[index] = StringOps::atoi(parameters[index]);
        }

        if (step > time) {
            return true;
        }

        std::cout << "get(" << MPILayer().rank() << ") " << c << "\n";
        pipe.addSteeringFeedback("bingo bongo");

        if (validRegion.count(c)) {
            std::cout << "  get result(" << c << ") = " << accessor->get(grid->get(c)) << "\n";
            return true;
        }

        return false;
    }

private:
    AccessorPtr accessor;
};

}

}

#endif
