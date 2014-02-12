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
#include <string>


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
    Coord<3> origin(0,0.0);
    int dim = atoi(argv[1]);
    std::string scotch = "Scotch";
    std::string zCurve = "ZCurve";
    std::string recBi = "RecBi";
    std::string checker = "Checker";
    std::string ptscotch = "PTScotch";
    std::string striping = "Striping";
    std::string dimString = argv[1];
    Coord<3> dimensions(dim,dim,dim);
    std::vector<std::size_t> weights;
    std::ofstream outputScotch,outputZCurve,outputRecBi,outputCheck, outputPTScotch, outputStriping;
    outputScotch.open((dimString + scotch).c_str());
    outputZCurve.open((dimString + zCurve).c_str());
    outputRecBi.open((dimString + recBi).c_str());
    outputCheck.open((dimString + checker).c_str());
    outputPTScotch.open((dimString + ptscotch).c_str());
    outputStriping.open((dimString + striping).c_str());
    for(int i = 4; i <= 200; i+=2){
        std::cout << "Round: " << i << std::endl;
        int remain = (dim*dim*dim)%i;
        int weight = (dim*dim*dim)/i;
        for(int j = 1; j <= i; ++j){
            weights.push_back(weight);
        }
        weights[i-1] += remain;
        int sum = 0;
        for(unsigned int l = 0; l < weights.size(); ++l){
            sum += weights[l];
        }
        Partition<3> *scotch = new ScotchPartition<3>(origin, dimensions, 0, weights);
        output(scotch,outputScotch,weights.size());
        delete scotch;
        Partition<3> *zCurve = new ZCurvePartition<3>(origin, dimensions, 0, weights);
        output(zCurve,outputZCurve,weights.size());
        delete zCurve;
        Partition<3> *recBi = new RecursiveBisectionPartition<3>(origin, dimensions, 0, weights);
        output(recBi,outputRecBi,weights.size());
        delete recBi;
        Partition<3> *checker = new CheckerboardingPartition<3>(origin, dimensions, 0, weights);
        output(checker,outputCheck,weights.size());
        delete checker;

        Partition<3> *ptscotch = new PTScotchPartition<3>(origin, dimensions, 0, weights);
        output(ptscotch,outputPTScotch,weights.size());
        delete checker;

        Partition<3> *striping = new CheckerboardingPartition<3>(origin, dimensions, 0, weights);
        output(striping,outputStriping,weights.size());
        delete checker;

        weights.clear();
    }
    MPI_Finalize();
}
