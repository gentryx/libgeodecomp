#ifndef LIBGEODECOMP_LIBGEODECOMP_H
#define LIBGEODECOMP_LIBGEODECOMP_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_FEATURE_HPX
#include <hpx/config.hpp>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#endif

#ifdef LIBGEODECOMP_FEATURE_MPI
#include <mpi.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#endif

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/fixedarray.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#endif
