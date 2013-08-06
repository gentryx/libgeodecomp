#ifndef LIBGEODECOMP_TESTBED_PERFORMANCE_TESTS_EVALUATE_H
#define LIBGEODECOMP_TESTBED_PERFORMANCE_TESTS_EVALUATE_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/stringops.h>

extern std::string revision;

template<class BENCHMARK>
void evaluate(BENCHMARK benchmark, const LibGeoDecomp::Coord<3>& dim)
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    std::stringstream buf;
    buf << now;
    std::string nowString = buf.str();
    nowString.resize(20);

    int hostnameLength = 2048;
    std::string hostname(hostnameLength, ' ');
    gethostname(&hostname[0], hostnameLength);
    hostname = LibGeoDecomp::StringOps::tokenize(hostname, " ")[0];

    std::string device = benchmark.device();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << std::setw(18) << revision << "; "
              << nowString << " ; "
              << std::setw(17) << hostname << "; "
              << std::setw(48) << device << "; "
              << std::setw( 8) << benchmark.order() <<  "; "
              << std::setw(16) << benchmark.family() <<  "; "
              << std::setw( 8) << benchmark.species() <<  "; "
              << std::setw(24) << dim <<  "; "
              << std::setw(12) << benchmark.performance(dim) <<  "; "
              << std::setw( 8) << benchmark.unit() <<  "\n";
}

#endif
