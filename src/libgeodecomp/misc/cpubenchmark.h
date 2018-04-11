#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H

#include <libflatarray/testbed/cpu_benchmark.hpp>

namespace LibGeoDecomp {

/**
 * Base class for all performance tests that run on the CPU (as
 * opposed to the GPU).
 */
class CPUBenchmark : public LibFlatArray::cpu_benchmark
{};

}

#endif
