#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/hindexingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/hilbertpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class PartitionBenchmarkTest : public CxxTest::TestSuite 
{
public:
    
    void testLargeBenchmark() 
    {
//         benchmark<HIndexingPartition, HIndexingPartition::Iterator>("HIndexing");
//         benchmark<StripingPartition,  StripingPartition::Iterator >("Striping ");
//         benchmark<HilbertPartition,   HilbertPartition::Iterator  >("Hilbert  ");
//         benchmark<ZCurvePartition,    ZCurvePartition::Iterator   >("ZCurve   ");
    }

private:
    template<class PARTITION, class ITERATOR>
    void benchmark(const std::string& label)
    {
        Coord<2> dim(32*1024, 32*1024);
        Coord<2> accu;

        long long t1 = Chronometer::timeUSec();
        PARTITION h(Coord<2>(100, 200), dim);
        ITERATOR end = h.end();
        for (ITERATOR i = h.begin(); i != end; ++i) 
            accu += *i;
        long long t2 = Chronometer::timeUSec();

        std::cout << "accu == " << accu << "\n";
        double span = (t2-t1) / 1000.0;
        std::cout << label << ": " << span << "\n";            
    }

};

}
}
