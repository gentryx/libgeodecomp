// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/geometry/partitions/hilbertpartition.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

HilbertPartition::Form HilbertPartition::squareFormTransitions[4][4] = {
    {LL_TO_UL, LL_TO_LR, LL_TO_LR, UR_TO_LR}, // LL_TO_LR
    {LL_TO_LR, LL_TO_UL, LL_TO_UL, UR_TO_UL}, // LL_TO_UL
    {UR_TO_UL, UR_TO_LR, UR_TO_LR, LL_TO_LR}, // UR_TO_LR
    {UR_TO_LR, UR_TO_UL, UR_TO_UL, LL_TO_UL}  // UR_TO_UL
};

// (sub-)sectors:
// 01
// 23
int HilbertPartition::squareSectorTransitions[4][4] = {
    {2, 0, 1, 3}, // LL_TO_LR
    {2, 3, 1, 0}, // LL_TO_UL
    {1, 0, 2, 3}, // UR_TO_LR
    {1, 3, 2, 0}  // UR_TO_UL
};

SharedPtr<HilbertPartition::CacheType>::Type HilbertPartition::squareCoordsCache;
Coord<2> HilbertPartition::maxCachedDimensions;
bool HilbertPartition::cachesInitialized = HilbertPartition::fillCaches();

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
