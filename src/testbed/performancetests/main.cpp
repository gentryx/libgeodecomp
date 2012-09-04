#include <boost/date_time/posix_time/posix_time.hpp>
#include <iomanip>
#include <iostream>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/region.h>
#include <unistd.h>

using namespace LibGeoDecomp;

std::string revision;

class RegionInsert
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "RegionInsert";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(Coord<3> dim)
    {
        long long tStart = Chronometer::timeUSec();

        Region<3> r;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionIntersect
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "RegionIntersect";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(Coord<3> dim)
    {
        long long tStart = Chronometer::timeUSec();

        Region<3> r1;
        Region<3> r2;

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        for (int z = 1; z < (dim.z() - 1); ++z) {
            for (int y = 1; y < (dim.y() - 1); ++y) {
                r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
            }
        }

        Region<3> r3 = r1 & r2;

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

template<class BENCHMARK>
void evaluate(BENCHMARK benchmark, const Coord<3>& dim)
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    std::stringstream buf;
    buf << now;
    std::string nowString = buf.str();
    nowString.resize(20);

    int hostnameLength = 2048;
    std::string hostname(hostnameLength, ' ');
    gethostname(&hostname[0], hostnameLength);
    int actualLength = 0;
    for (int i = 0; i < hostnameLength; ++i) {
        if (hostname[i] == 0) {
            actualLength = i;
        }
    }
    hostname.resize(actualLength);

    FILE *output = popen("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -c 14-", "r");
    int idLength = 2048;
    std::string cpuID(idLength, ' ');
    idLength = fread(&cpuID[0], 1, idLength, output);
    cpuID.resize(idLength - 1);
    pclose(output);
    

    std::cout << std::setiosflags(std::ios::left);
    std::cout << std::setw(5) << revision << "; " 
              << nowString << " ; " 
              << std::setw(35) << hostname << "; " 
              << std::setw(48) << cpuID << "; " 
              << std::setw( 8) << benchmark.order() <<  "; " 
              << std::setw(16) << benchmark.family() <<  "; " 
              << std::setw( 8) << benchmark.species() <<  "; " 
              << std::setw(20) << dim <<  "; " 
              << std::setw(10) << benchmark.performance(dim) <<  "; " 
              << std::setw( 8) << benchmark.unit() <<  "\n";
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " REVISION\n";
        return 1;
    }

    revision = argv[1];

    std::cout << "#rev ; date                 ; host                               ; device                                          ; order   ; family          ; species ; dimensions          ; perf      ; unit\n";

    evaluate(RegionInsert(), Coord<3>( 128,  128,  128));
    evaluate(RegionInsert(), Coord<3>( 512,  512,  512));
    evaluate(RegionInsert(), Coord<3>(2048, 2048, 2048));

    evaluate(RegionIntersect(), Coord<3>( 128,  128,  128));
    evaluate(RegionIntersect(), Coord<3>( 512,  512,  512));
    evaluate(RegionIntersect(), Coord<3>(2048, 2048, 2048));

    return 0;
}
