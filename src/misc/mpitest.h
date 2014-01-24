#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI
#include <mpi.h>
#include <cxxtest/GlobalFixture.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/typemaps.h>

/**
 * OpenMPI can't run multiple MPI_Init() - MPI_Finalize() cycles
 * during one program execution. Therefore this file has to hook these
 * calls into CxxTest's global fixtures (so the get executed exactly
 * once).
 */
class MPIGlobalFixture : public CxxTest::GlobalFixture
{
    virtual bool setUpWorld()
    {
        int argc = 0;
        char **argv = 0;
        MPI_Init(&argc, &argv);
        LibGeoDecomp::Typemaps::initializeMaps();
        return true;
    }

    virtual bool tearDownWorld()
    {
        MPI_Finalize();
        return true;
    }
};

/**
 * This test suite ensures that MPI has been successfully
 * initialized. Don't delete it as then the global fixtures above
 * won't be executed. (urgh, now THAT's ugly!)
 */
class MPITest : public CxxTest::TestSuite
{
public:

    void testMPIUpAndRunning()
    {
        TS_ASSERT_THROWS_NOTHING(  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank); );
    }
};

static MPIGlobalFixture mpiSetupAndFinalizer;

#endif
