#ifndef LIBGEODECOMP_LIBGEODECOMP_H
#define LIBGEODECOMP_LIBGEODECOMP_H
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_FEATURE_HPX
#include <hpx/config.hpp>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#endif

#ifdef LIBGEODECOMP_FEATURE_MPI
#include <mpi.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#endif

#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/storage/fixedarray.h>

#endif
