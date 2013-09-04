#include "typemaps.h"
#include <algorithm>

namespace MPI {
    CLASS_VARS
}

NAMESPACE_BEGIN

// Member Specification, holds all relevant information for a given member.
class MemberSpec
{
public:
    MemberSpec(MPI_Aint address, MPI_Datatype type, int length) :
        address(address),
        type(type),
        length(length)
    {}

    MPI_Aint address;
    MPI_Datatype type;
    int length;
};

bool addressLower(MemberSpec a, MemberSpec b)
{
    return a.address < b.address;
}

METHOD_DEFINITIONS
void Typemaps::initializeMaps()
{
    ASSIGNMENTS
}

NAMESPACE_END
