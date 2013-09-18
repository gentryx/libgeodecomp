#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H

#include <libgeodecomp/testbed/performancetests/benchmark.h>

class CPUBenchmark : public Benchmark
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string device()
    {
        FILE *output = popen("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -c 14-", "r");
        int idLength = 2048;
        std::string cpuID(idLength, ' ');
        idLength = fread(&cpuID[0], 1, idLength, output);
        cpuID.resize(idLength - 1);
        pclose(output);
        return cpuID;
    }
};

#endif
