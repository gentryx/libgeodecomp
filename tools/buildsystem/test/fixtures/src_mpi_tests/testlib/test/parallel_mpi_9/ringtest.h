#include <cxxtest/TestSuite.h>
#include <iostream>
#include <mpi.h>

class RingTest : public CxxTest::TestSuite 
{
public:
    
    void testSimple() {
        MPI::Comm& comm = MPI::COMM_WORLD;
        int rank = comm.Get_rank();
        int size = comm.Get_size();
        int prev = (rank - 1 + size) % size;
        int next = (rank + 1) % size;

        int sum = 0;
        int val = 1 << rank;

        if (rank == 0) {
            comm.Send(&val, 1, MPI::INT, next, 4711);
            comm.Recv(&sum, 1, MPI::INT, prev, 4711);
            std::cout << "size is " << size << "\n";
            std::cout << "sum is " << sum << "\n";
        } else {            
            comm.Recv(&sum, 1, MPI::INT, prev, 4711);
            sum += val;
            comm.Send(&sum, 1, MPI::INT, next, 4711);
        }
    }
};
