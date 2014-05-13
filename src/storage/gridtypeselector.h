#ifndef LIBGEODECOMP_STORAGE_GRIDTYPESELECTOR_H
#define LIBGEODECOMP_STORAGE_GRIDTYPESELECTOR_H

#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/soagrid.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT, typename SUPPORTS_SOA>
class GridTypeSelector;

template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT, APITraits::FalseType>
{
public:
    typedef DisplacedGrid<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT> Value;
};

template<typename CELL_TYPE, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT, APITraits::TrueType>
{
public:
    typedef SoAGrid<CELL_TYPE, TOPOLOGY, TOPOLOGICALLY_CORRECT> Value;
};

}

#endif
