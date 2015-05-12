#ifndef LIBGEODECOMP_STORAGE_GRIDTYPESELECTOR_H
#define LIBGEODECOMP_STORAGE_GRIDTYPESELECTOR_H

#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/apitraits.h>

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

template<typename CELL_TYPE, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TopologiesHelpers::UnstructuredTopology,
                       TOPOLOGICALLY_CORRECT, APITraits::FalseType>
{
private:
    typedef typename APITraits::SelectSellType<CELL_TYPE>::Value ValueType;
    static const std::size_t MATRICES = APITraits::SelectSellMatrices<CELL_TYPE>::VALUE;
    static const int C = APITraits::SelectSellC<CELL_TYPE>::VALUE;
    static const int SIGMA = APITraits::SelectSellSigma<CELL_TYPE>::VALUE;
public:
    typedef UnstructuredGrid<CELL_TYPE, MATRICES, ValueType, C, SIGMA> Value;
};

template<typename CELL_TYPE, bool TOPOLOGICALLY_CORRECT>
class GridTypeSelector<CELL_TYPE, TopologiesHelpers::UnstructuredTopology,
                       TOPOLOGICALLY_CORRECT, APITraits::TrueType>
{
private:
    typedef typename APITraits::SelectSellType<CELL_TYPE>::Value ValueType;
    static const std::size_t MATRICES = APITraits::SelectSellMatrices<CELL_TYPE>::VALUE;
    static const int C = APITraits::SelectSellC<CELL_TYPE>::VALUE;
    static const int SIGMA = APITraits::SelectSellSigma<CELL_TYPE>::VALUE;
public:
    typedef UnstructuredGridSoA<CELL_TYPE, MATRICES, ValueType, C, SIGMA> Value;
};

}

#endif
