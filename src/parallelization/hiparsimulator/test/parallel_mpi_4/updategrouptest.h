#include <fstream>
#include <iostream>

#include <libgeodecomp/io/simplecellinitializer.h>
#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    // typedef ZCurvePartition Partition;

    void testBench()
    {
//         int width, height, maxSteps, ghostZoneWidth;
//         std::ifstream params("./params");
//         if (!params)
//             throw std::logic_error("could not open param file");
//         params >> width;
//         params >> height;
//         params >> maxSteps;
//         params >> ghostZoneWidth;
//         if (MPILayer().rank() == 0) {
//             std::cout << "width: " << width << "\n"
//                       << "height: " << height << "\n"
//                       << "maxSteps: " << maxSteps << "\n"
//                       << "ghostZoneWidth: " << ghostZoneWidth << "\n";
//         }
//         SimpleInitializer init(width, height, maxSteps);   
//         Partition partition(Coord(0, 0), Coord(width, height));
//         SuperVector<unsigned> weights = genWeights(width, height, MPILayer().size());
//         UpdateGroup<SimpleCell, Partition, RegionAccumulator> group(
//             partition,
//             weights,
//             0,
//             CoordRectangle(Coord(0, 0), Coord(width, height)),
//             ghostZoneWidth,
//             &init,
//             0,
//             0);

//         if (MPILayer().rank() == 0) 
//             std::cout << "weights: " << weights << "\n";

//         long long tStart = Chronometer::timeUSec();
//         group.nanoStep(maxSteps, 1);
//         long long tEnd = Chronometer::timeUSec();

//         if (MPILayer().rank() == 0) 
//             std::cout << "wallclock time: " << (tEnd - tStart) << " res: " << (*group.getGrid())[Coord(10, 10)].val << "\n";
    }
    
private:

    SuperVector<unsigned> genWeights(const unsigned& width, const unsigned& height, const unsigned& size)
    {
        SuperVector<unsigned> ret(size);
        unsigned totalSize = width * height;
        for (int i = 0; i < ret.size(); ++i) 
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        return ret;            
    }

    unsigned pos(const unsigned& i, const unsigned& size, const unsigned& totalSize)
    {
        return i * totalSize / size;
    }
};

}
}
