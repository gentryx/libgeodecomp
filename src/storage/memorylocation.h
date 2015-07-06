#ifndef LIBGEODECOMP_STORAGE_MEMORYLOCATION_H
#define LIBGEODECOMP_STORAGE_MEMORYLOCATION_H

namespace LibGeoDecomp {

/**
 * Enumeration which allows us to distinguish to which address space a
 * given pointer belongs.
 */
class MemoryLocation
{
public:
    enum Location {HOST=0, CUDA_DEVICE=1};
};

}

#endif
