#include <iostream>
#include <mpi.h>
#include "car.h"
#include "bmw.h"
#include "typemaps.h"

/**
 * This simple demo program shows how to use the Typemaps class, which
 * provides the MPI datatypes created from the given class hierarchy
 * by the TypemapGenerator.
 */
int main(int argc, char *argv[])
{
    MPI::Init(argc, argv);
    Typemaps::initializeMaps();

    const int NUM_CARS = 128;
    Car alphas[NUM_CARS];
    Car romeos[NUM_CARS];
    int tag = 4711;
    MPI::Request requests[2];

    std::cout << "sending...\n";
    requests[0] = MPI::COMM_WORLD.Isend(&alphas, NUM_CARS, MPI::CAR, 0, tag);
    std::cout << "receiving...\n";
    requests[1] = MPI::COMM_WORLD.Irecv(&romeos, NUM_CARS, MPI::CAR, 0, tag);
    MPI::Request::Waitall(2, requests);
    std::cout << "done.\n";

    MPI::Finalize();
    return 0;
}
