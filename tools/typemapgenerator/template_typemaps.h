#ifndef NAMESPACE_GUARD_H
#define NAMESPACE_GUARD_H

#include <mpi.h>
#include <complex>
HEADERS

CLASS_VARS

NAMESPACE_BEGIN
class Typemaps
{
public:
    static void initializeMaps();

    template<typename T>
    static inline MPI_Datatype lookup()
    {
        return lookup((T*)0);
    }

private:
    template<typename T>
    static MPI_Aint getAddress(T *address)
    {
        MPI_Aint ret;
        MPI_Get_address(address, &ret);
        return ret;
    }

    MAPGEN_DECLARATIONS

public:
    LOOKUP_DEFINITIONS
};
NAMESPACE_END

#endif
