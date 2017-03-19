#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include "hpxserialization.h"

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::Chronometer)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::Color)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::Coord<1 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::Coord<2 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::Coord<3 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::FloatCoord<1 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::FloatCoord<2 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::FloatCoord<3 >)
HPX_SERIALIZATION_REGISTER_CLASS(LibGeoDecomp::NonPoDTestCell)


#endif
