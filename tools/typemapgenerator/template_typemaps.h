#ifndef NAMESPACE_GUARDTYPEMAPS_H
#define NAMESPACE_GUARDTYPEMAPS_H

#include <mpi.h>
#include <complex>
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
