#ifndef LIBGEODECOMP_EXAMPLES_GAMEOFLIFE_ADCIRC_HULL_H
#define LIBGEODECOMP_EXAMPLES_GAMEOFLIFE_ADCIRC_HULL_H

#include <libgeodecomp/geometry/floatcoord.h>
#include <vector>

using namespace LibGeoDecomp;

std::vector<FloatCoord<2> > convexHull(std::vector<FloatCoord<2> > *points);

#endif
