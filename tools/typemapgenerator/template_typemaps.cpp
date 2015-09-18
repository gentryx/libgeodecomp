#include "typemaps.h"
#include <algorithm>
#include <stdexcept>

CLASS_VARS

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
    if (mapsCreated) {
        throw std::logic_error("Typemaps already initialized, duplicate initialization would leak memory");
    }

    if (sizeof(std::size_t) != sizeof(unsigned long)) {
        throw std::logic_error("MPI_UNSIGNED_LONG not suited for communication of std::size_t, needs to be redefined");
    }

    int mpiInitState = 0;
    MPI_Initialized(&mpiInitState);
    if (!mpiInitState) {
        throw std::logic_error("MPI needs to be initialized prior to setting up Typemaps");
    }

    ASSIGNMENTS

    mapsCreated = true;
}

bool Typemaps::mapsCreated = false;

NAMESPACE_END
