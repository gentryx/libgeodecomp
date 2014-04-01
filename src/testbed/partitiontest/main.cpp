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
    int dimx, dimy, dimz;
    if(argc == 2){
        dimx = atoi(argv[1]);
        dimy = dimx;
        dimz = dimx;
    } else {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
        dimz = atoi(argv[3]);
    }
    std::string scotch = "Scotch";
    std::string zCurve = "ZCurve";
    std::string recBi = "RecBi";
    std::string checker = "Checker";
    std::string ptscotch = "PTScotch";
    std::string striping = "Striping";
    std::string dimString = argv[1];
    Coord<3> dimensions(dimx,dimy,dimz);
    std::vector<std::size_t> weights;
    std::ofstream outputScotch,outputZCurve,outputRecBi,outputCheck, outputPTScotch, outputStriping;
    outputScotch.open((dimString + 'x' + dimString + scotch).c_str());
    outputZCurve.open((dimString + 'x' + dimString + zCurve).c_str());
    outputRecBi.open((dimString + 'x' + dimString + recBi).c_str());
    outputCheck.open((dimString + 'x' + dimString + checker).c_str());
    outputPTScotch.open((dimString + 'x' + dimString + ptscotch).c_str());
    outputStriping.open((dimString + 'x' + dimString + striping).c_str());
    for(int i = 4; i <= 200; ++i){
        std::cout << "Round: " << i << std::endl;
        int remain = (dimx*dimy*dimz)%i;
        int weight = (dimx*dimy*dimz)/i;

        for(int j = 1; j <= i; ++j){
            weights.push_back(weight);
        }
        weights[i-1] += remain;
        /*int sum = 0;
        for(unsigned int l = 0; l < weights.size(); ++l){
            sum += weights[l];
            }*/
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                std::cout << "scotch" << std::endl;
                Partition<3> *scotch = new ScotchPartition<3>(origin, dimensions, 0, weights);
                output(scotch,outputScotch,weights.size());
                delete scotch;
            }
            #pragma omp section
            {
                std::cout << "ZCurve" << std::endl;
                Partition<3> *zCurve = new ZCurvePartition<3>(origin, dimensions, 0, weights);
                output(zCurve,outputZCurve,weights.size());
                delete zCurve;
            }
            #pragma omp section
            {
                std::cout << "recBi" << std::endl;
                Partition<3> *recBi = new RecursiveBisectionPartition<3>(origin, dimensions, 0, weights);
                output(recBi,outputRecBi,weights.size());
                delete recBi;
            }
            #pragma omp section
            {
                std::cout << "checker" << std::endl;
                Partition<3> *checker = new CheckerboardingPartition<3>(origin, dimensions, 0, weights);
                output(checker,outputCheck,weights.size());
                delete checker;
            }
            #pragma omp section
            {
                std::cout << "ptscotch" << std::endl;
                Partition<3> *ptscotch = new PTScotchPartition<3>(origin, dimensions, 0, weights);
                output(ptscotch,outputPTScotch,weights.size());
                delete ptscotch;
            }
            #pragma omp section
            {
                std::cout << "striping" << std::endl;
                Partition<3> *striping = new CheckerboardingPartition<3>(origin, dimensions, 0, weights);
                output(striping,outputStriping,weights.size());
                delete striping;
            }
        }
        weights.clear();
    }
    MPI_Finalize();
}
