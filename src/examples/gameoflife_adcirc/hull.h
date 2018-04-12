#ifndef LIBGEODECOMP_EXAMPLES_GAMEOFLIFE_ADCIRC_HULL_H
#define LIBGEODECOMP_EXAMPLES_GAMEOFLIFE_ADCIRC_HULL_H

#include <libgeodecomp/geometry/floatcoord.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

using namespace LibGeoDecomp;

std::vector<FloatCoord<2> > convexHull(std::vector<FloatCoord<2> > *points);

#endif
