#include <mpi.h>
#include <libgeodecomp/geometry/partitions/scotchpartition.h>
#include <libgeodecomp/geometry/partitions/ptscotchpartition.h>
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <iostream>
#include <fstream>


using namespace LibGeoDecomp;


void print(Region<3> reg){
    std::cout << reg.size() <<
        " " << reg.expand(1).size() - reg.size() << " Ratio: "
              << ((float)reg.expand(1).size() - reg.size()) / reg.size() <<std::endl;
}

void print(Region<2> reg){
    std::cout << reg.size() <<
        " " << reg.expand(1).size() - reg.size() << " Ratio: "
              << ((float)reg.expand(1).size() - reg.size()) / reg.size() <<std::endl;
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    Coord<3> origin(0,0);
    int dim = atoi(argv[1]);
    Coord<3> dimensions(dim,dim,dim);
    std::vector<std::size_t> weights;
    std::ofstream outputScotch,outputZCurve;
    outputScotch.open("outputScotch.txt");
    outputZCurve.open("outputZCurve.txt");
    for(int i = 4; i <= 200; i+=4){
        std::cout << i << std::endl;
        int remain = (dim*dim*dim)%i;
        int weight = (dim*dim*dim)/i;
        for(int j = 1; j <= i; ++j){
            std::cout << "weight: " << weight << std::endl;
            weights.push_back(weight);
        }
        weights[i-1] += remain;
        int sum = 0;
        for(unsigned int l = 0; l < weights.size(); ++l){
            std::cout << l << ": " << weights[l] << " ";
            sum += weights[l];
        }
        std::cout << "sum: " <<  sum << " " << weights.size() << std::endl;
        weights << 10 << 10;
        std::cout << "Scotch" << std::endl;
        ScotchPartition<3> scotch(origin, dimensions, 0, weights);
        Region<3> regScotch;
        int min,max,avg,tmp;
        min = dim*dim*dim;
        max = 0;
        avg = 0;
        for(unsigned int j = 0;j < weights.size(); ++j){
            regScotch = scotch.getRegion(j);
            tmp = regScotch.expand(1).size() - regScotch.size();
            if(tmp < min){
                min = tmp;
            }
            if(tmp > max){
                max = tmp;
            }
            avg += tmp;
        }
        avg /= weights.size();
        outputScotch << i << "\t" << min << "\t" << avg << "\t" << max << "\n";
        std::cout << "RecBi" << std::endl;
        ZCurvePartition<3> zCurve(origin, dimensions, 0, weights);
        Region<3> regzCurve;
        min = dim*dim*dim;
        max = 0;
        avg = 0;
        for(unsigned int j = 0;j < weights.size(); ++j){
            regzCurve = zCurve.getRegion(j);
            tmp = regzCurve.expand(1).size() - regzCurve.size();
            if(tmp < min){
                min = tmp;
            }
            if(tmp > max){
                max = tmp;
            }
            avg += tmp;
        }
        avg /= weights.size();
        outputZCurve << i << "\t" << min << "\t" << avg << "\t" << max << "\n";
        weights.clear();
    }
    /*ScotchPartition<3> sp(origin,dimensions, 0, weights);
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
        }*/
    MPI_Finalize();
}
