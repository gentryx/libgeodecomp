#include <mpi.h>
#include <libgeodecomp/geometry/partitions/scotchpartition.h>
#include <libgeodecomp/geometry/partitions/ptscotchpartition.h>
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>



using namespace LibGeoDecomp;


void print(Region<3> reg){
    std::cout << reg.size() <<
        " " << reg.expand(1).size() - reg.size() << " Ratio: "
              << ((float)reg.expand(1).size() - reg.size()) / reg.size() <<std::endl;
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    Coord<3> origin(0,0);
    Coord<3> dimensions(120,128,100);
    std::vector<std::size_t> weights;
    weights << 999 << 999 << 120*128*100-999*2;
    ScotchPartition<3> sp(origin,dimensions, 0, weights);
    PTScotchPartition<3> ptp(origin,dimensions, 0, weights);
    CheckerboardingPartition<3> cp(origin,dimensions, 0, weights);
    ZCurvePartition<3> zp(origin,dimensions, 0, weights);
    StripingPartition<3> stripingP(origin,dimensions, 0, weights);
    RecursiveBisectionPartition<3> rP(origin,dimensions, 0, weights);
    Region<3> regP;
    Region<3> regPTS;
    Region<3> regC;
    Region<3> regZ;
    Region<3> regStrip;
    Region<3> regRec;
    for(unsigned int i = 0;i < weights.size(); ++i){
        regP = sp.getRegion(i);
        regPTS = ptp.getRegion(i);
        regC = cp.getRegion(i);
        regZ = zp.getRegion(i);
        regStrip = stripingP.getRegion(i);
        regRec = rP.getRegion(i);
        std::cout << "Scotch: ";
        print(regP);
        std::cout << "PTScotch";
        print(regPTS);
        std::cout << "Checkerboarding: ";
        print(regC);
        std::cout << "ZCurve: ";
        print(regZ);
        std::cout << "Striping: ";
        print(regStrip);
        std::cout << "RecursiveBisection: ";
        print(regRec);
        std::cout << std::endl;
        }
    MPI_Finalize();
}
