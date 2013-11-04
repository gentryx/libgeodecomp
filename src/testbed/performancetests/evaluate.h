#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_EVALUATE_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_EVALUATE_H

#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/stringops.h>

class Evaluate
{
public:
     Evaluate(const std::string& revision) :
        revision(revision)
    {}

    void printHeader()
    {
        std::cout << "#rev              ; date                 ; host            ; device                                          ; order   ; family                          ; species ; dimensions              ; perf        ; unit" << std::endl;
    }

    template<class BENCHMARK>
    void operator()(BENCHMARK benchmark, const LibGeoDecomp::Coord<3>& dim, bool output = true)
    {
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        std::stringstream buf;
        buf << now;
        std::string nowString = buf.str();
        nowString.resize(20);

        std::string device = benchmark.device();

        int hostnameLength = 2048;
        std::string hostname(hostnameLength, ' ');
        gethostname(&hostname[0], hostnameLength);
        // cuts string at first 0 byte, required as gethostname returns 0-terminated strings
        hostname = std::string(hostname.c_str());

        double performance = benchmark.performance(dim);

        if (output) {
            std::cout << std::setiosflags(std::ios::left);
            std::cout << std::setw(18) << revision << "; "
                      << nowString << " ; "
                      << std::setw(16) << hostname << "; "
                      << std::setw(48) << device << "; "
                      << std::setw( 8) << benchmark.order() <<  "; "
                      << std::setw(32) << benchmark.family() <<  "; "
                      << std::setw( 8) << benchmark.species() <<  "; "
                      << std::setw(24) << dim <<  "; "
                      << std::setw(12) << performance <<  "; "
                      << std::setw( 8) << benchmark.unit() << std::endl;
        }
    }

private:
    std::string revision;
};

#endif
