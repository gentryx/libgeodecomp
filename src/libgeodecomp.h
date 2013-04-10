#ifndef LIBGEODECOMP_LIBGEODECOMP_H
#define LIBGEODECOMP_LIBGEODECOMP_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_FEATURE_MPI
#include <mpi.h>
#endif

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/misc/fixedarray.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#endif
