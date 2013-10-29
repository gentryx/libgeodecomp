#ifndef LIBGEODECOMP_IO_REMOTESTEERER_GETACTION_H
#define LIBGEODECOMP_IO_REMOTESTEERER_GETACTION_H

#include <libgeodecomp/io/remotesteerer/passthroughaction.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

template<typename CELL_TYPE>
class GetAction : public PassThroughAction<CELL_TYPE>
{
public:
    GetAction(std::string memberName) :
        PassThroughAction<CELL_TYPE>(
            "get_" + memberName,
            "usage: \"get_" + memberName + " T X Y [Z] MEMBER\", will return member MEMBER at time step T of cell at grid coordinate (X, Y, Z) if the model is 3D, or (X, Y) in the 2D case")
    {}
};

}

}

#endif
