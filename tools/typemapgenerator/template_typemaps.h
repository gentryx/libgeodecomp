#ifndef _NAMESPACE_GUARDtypemaps_h_
#define _NAMESPACE_GUARDtypemaps_h_

#include <complex>
#include <mpi.h>
HEADERS

namespace MPI {
    CLASS_VARS
}

NAMESPACE_BEGIN
class Typemaps {
public:
    static void initializeMaps();

    template<typename T>
    static inline MPI::Datatype lookup() {
        return lookup((T*)0);
    }

private:
    MAPGEN_DECLARATIONS

    LOOKUP_DEFINITIONS
};
NAMESPACE_END

#endif
