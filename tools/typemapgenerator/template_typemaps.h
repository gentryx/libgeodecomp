#ifndef NAMESPACE_GUARDTYPEMAPS_H
#define NAMESPACE_GUARDTYPEMAPS_H

#include <mpi.h>
#include <complex>
HEADERS

CLASS_VARS

NAMESPACE_BEGIN
/**
 * Utility class which can set up and yield MPI datatypes for custom datatypes.
 *
 * AUTO-GENERATED CODE. DO NOT EDIT. CHANGES WILL BE LOST. Refer to
 * typemapgenerator for further reference.
 */
class Typemaps
{
public:
    /**
     * Sets up MPI datatypes for all registered types.
     */
    static void initializeMaps();

    /**
     * Avoids duplicate initialization
     */
    static inline void initializeMapsIfUninitialized()
    {
        if (!initialized()) {
            initializeMaps();
        }
    }

    /**
     * Query initialization state
     */
    static inline bool initialized()
    {
        return mapsCreated;
    }

    /**
     * Performs an internal lookup. Works for custom, registered types
     * and for built-in types (char, int, size_t...). Compilation will
     * fail for unknown types.
     */
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

    static bool mapsCreated;

    MAPGEN_DECLARATIONS

public:
    LOOKUP_DEFINITIONS
};
NAMESPACE_END

#endif
