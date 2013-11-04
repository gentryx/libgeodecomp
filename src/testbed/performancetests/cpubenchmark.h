#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H

#include <fstream>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/testbed/performancetests/benchmark.h>

namespace LibGeoDecomp {

class CPUBenchmark : public Benchmark
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string device()
    {
        std::ifstream file("/proc/cpuinfo");
        size_t bufferSize = 2048;
        std::string buffer(bufferSize, ' ');
        while (file.getline(&buffer[0], bufferSize)) {
            std::vector<std::string> tokens = StringOps::tokenize(buffer, ":");
            std::vector<std::string> fields = StringOps::tokenize(tokens[0], " \t");

            if ((fields[0] == "model") && (fields[1] == "name")) {
                return StringOps::join(StringOps::tokenize(tokens[1], " \t"), " ");
            }
        }

        throw std::runtime_error("could not parse /proc/cpuinfo");
    }
};

}

#endif
