#ifndef LIBGEODECOMP_STORAGE_CUDAGRIDTYPESELECTOR_H
#define LIBGEODECOMP_STORAGE_CUDAGRIDTYPESELECTOR_H

#include <libgeodecomp/config.h>

#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/cudasoagrid.h>

namespace LibGeoDecomp {

/**
 * Similar to GridTypeSelector, this code helps with selecting the
 * correct grid type with regard to a CELL_TYPE's traits -- only for
 * CUDA codes.
 */
template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT, typename SUPPORTS_SOA>
class CUDAGridTypeSelector;

/**
 * see above.
 */
template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT, APITraits::FalseType>
{
public:
    typedef CUDAGrid<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT> Value;
};

/**
 * see above.
 */
template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT, APITraits::TrueType>
{
public:
    typedef CUDASoAGrid<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT> Value;
};

// fixme: missing CUDAUnstructured(SoA)Grid


}

#endif
