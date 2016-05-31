#ifndef LIBGEODECOMP_LIBGEODECOMP_H
#define LIBGEODECOMP_LIBGEODECOMP_H
#include <libgeodecomp/config.h>

#if defined (LIBGEODECOMP_WITH_HPX) || defined (LIBGEODECOMP_WITH_MPI)
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#endif

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#endif

#ifdef LIBGEODECOMP_WITH_MPI
#include <mpi.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#endif

#ifdef LIBGEODECOMP_WITH_VISIT
#include <libgeodecomp/io/visitwriter.h>
#endif

#include <libgeodecomp/communication/boostserialization.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/voronoimesher.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/serialbovwriter.h>
#include <libgeodecomp/io/silowriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/storage/boxcell.h>
#include <libgeodecomp/storage/containercell.h>
#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/storage/simplearrayfilter.h>
#include <libgeodecomp/storage/simplefilter.h>

#endif
