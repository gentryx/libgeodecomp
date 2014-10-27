#ifndef HULL_H
#define HULL_H

#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/image.h>

using namespace LibGeoDecomp;

double cross(const FloatCoord<2> &o, const FloatCoord<2> &a, const FloatCoord<2> &b);

bool floatCoordCompare(const FloatCoord<2> &a, const FloatCoord<2> &b);

std::vector<FloatCoord<2> > convexHull(std::vector<FloatCoord<2> > *points);

#endif
