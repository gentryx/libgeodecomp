#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/parallelization/mocksimulator.h>

/**
 * This hack is required when compiling with IBM's xlc++ on BG/Q
 * (juqueen), as this compiler is easily confused by templates.
 */
#ifdef __IBM_ATTRIBUTES
typedef LibGeoDecomp::Grid<LibGeoDecomp::TestCell<2,LibGeoDecomp::Stencils::Moore<2,1>,LibGeoDecomp::TopologiesHelpers::Topology<2,0,0,0>,LibGeoDecomp::TestCellHelpers::EmptyAPI,LibGeoDecomp::TestCellHelpers::StdOutput>,LibGeoDecomp::TopologiesHelpers::Topology<2,0,0,0> > TestGrid;
TestGrid dummy(LibGeoDecomp::Coord<2>(10, 20));
#endif

namespace LibGeoDecomp {

std::string MockSimulator::events;

}

#endif
