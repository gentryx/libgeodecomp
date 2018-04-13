// #include <libgeodecomp/parallelization/mocksimulator.h>

// /**
//  * This hack is required when compiling with IBM's xlc++ on BG/Q
//  * (juqueen), as this compiler is easily confused by templates.
//  */
// #ifdef __IBM_ATTRIBUTES
// typedef LibGeoDecomp::Grid<LibGeoDecomp::TestCell<2,LibGeoDecomp::Stencils::Moore<2,1>,LibGeoDecomp::TopologiesHelpers::Topology<2,0,0,0>,LibGeoDecomp::TestCellHelpers::EmptyAPI,LibGeoDecomp::TestCellHelpers::StdOutput>,LibGeoDecomp::TopologiesHelpers::Topology<2,0,0,0> > TestGrid;
// TestGrid dummy(LibGeoDecomp::Coord<2>(10, 20));
// #endif


// Kill some warnings in system headers:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4548 4710 4711 4996 )
#endif

#include <string>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

class MockSimulator {
public:
    static std::string MockSimulator::events;
};

std::string MockSimulator::events;

}

// #ifdef _MSC_BUILD
// #pragma warning( disable : 4710 )
// #endif
