#include <algorithm>
#include "typemaps.h"

namespace MPI {
    CLASS_VARS
}

NAMESPACE_BEGIN
// Member Specification, holds all relevant information for a given member.
class MemberSpec {
public:
    MemberSpec(MPI::Aint _address, MPI::Datatype _type, int _length) {
        address = _address;
        type = _type;
        length = _length;
    }

    MPI::Aint address;
    MPI::Datatype type;
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
