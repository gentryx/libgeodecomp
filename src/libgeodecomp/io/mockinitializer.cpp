// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/io/mockinitializer.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

std::string MockInitializer::events;

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
