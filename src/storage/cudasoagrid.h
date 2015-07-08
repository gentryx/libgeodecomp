#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libgeodecomp/storage/cudaarray.h>

namespace LibGeoDecomp {

/**
 * CUDA-enabled implementation of SoAGrid, relies on
 * LibFlatArray::soa_grid.
 */
template<typename CELL_TYPE,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDASoAGrid
{

};

}

#endif
