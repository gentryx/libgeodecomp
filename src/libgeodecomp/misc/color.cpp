// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/misc/color.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

const Color Color::BLACK(    0,   0  , 0);
const Color Color::WHITE(  255, 255, 255);

const Color Color::RED(    255,   0,   0);
const Color Color::GREEN(    0, 255,   0);
const Color Color::BLUE(     0,   0, 255);

const Color Color::CYAN(     0, 255, 255);
const Color Color::MAGENTA(255,   0, 255);
const Color Color::YELLOW( 255, 255,   0);

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
