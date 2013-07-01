#ifndef LIBGEODECOMP_IO_REMOTESTEERER_SETACTION
#define LIBGEODECOMP_IO_REMOTESTEERER_SETACTION

#include <libgeodecomp/io/remotesteerer/passthroughaction.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

template<typename CELL_TYPE>
class SetAction : public PassThroughAction<CELL_TYPE>
{
public:
    SetAction(std::string memberName) :
        PassThroughAction<CELL_TYPE>(
            "set" + memberName,
            "usage: \"get_" + memberName + " T X Y [Z] MEMBER VALUE\", will set member MEMBER at time step T of cell at grid coordinate (X, Y, Z) (if the model is 3D, or (X, Y) in the 2D case) to value VALUE")
    {}
};

}

}

#endif
