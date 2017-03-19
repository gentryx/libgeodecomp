// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/parallelization/mocksimulator.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
