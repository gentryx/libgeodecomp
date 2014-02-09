#include <mpi.h>
#include <libgeodecomp/geometry/partitions/scotchpartition.h>
#include <libgeodecomp/geometry/partitions/ptscotchpartition.h>
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <iostream>
#include <fstream>
#include <limits.h>


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

void output(Partition<3> * part, std::ofstream &output,unsigned int number){
    Region<3> reg;
    unsigned int min,max,avg,tmp;
    min = std::numeric_limits<int>::max();
    max = 0;
    avg = 0;
    for(unsigned int j = 0;j < number; ++j){
        reg = part->getRegion(j);
        tmp = reg.expand(1).size() - reg.size();
        if(tmp < min){
            min = tmp;
        }
        if(tmp > max){
            max = tmp;
        }
        avg += tmp;
    }
    avg /= number;
    output << number << "\t" << min << "\t" << avg << "\t" << max << "\n";
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    Coord<3> origin(0,0);
    int dim = atoi(argv[1]);
    Coord<3> dimensions(dim,dim,dim);
    std::vector<std::size_t> weights;
    std::ofstream outputScotch,outputZCurve,outputRecBi,outputCheck;
    outputScotch.open("outputScotch.txt");
    outputZCurve.open("outputZCurve.txt");
    outputRecBi.open("outputRecBi.txt");
    outputCheck.open("outputCheck.txt");
    for(int i = 4; i <= 200; i+=2){
        std::cout << "Round: " << i << std::endl;
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
        Partition<3> *scotch = new ScotchPartition<3>(origin, dimensions, 0, weights);
        output(scotch,outputScotch,weights.size());
        delete scotch;
        std::cout << "ZCurve" << std::endl;
        Partition<3> *zCurve = new ZCurvePartition<3>(origin, dimensions, 0, weights);
        output(zCurve,outputZCurve,weights.size());
        delete zCurve;
        std::cout << "RecBi" << std::endl;
        Partition<3> *recBi = new RecursiveBisectionPartition<3>(origin, dimensions, 0, weights);
        output(recBi,outputRecBi,weights.size());
        delete recBi;
        std::cout << "Checker" << std::endl;
        Partition<3> *checker = new CheckerboardingPartition<3>(origin, dimensions, 0, weights);
        output(checker,outputCheck,weights.size());
        delete checker;
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
