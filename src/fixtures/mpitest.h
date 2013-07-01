#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <mpi.h>
#include <cxxtest/GlobalFixture.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/mpilayer/typemaps.h>

/**
 * OpenMPI can't run multiple MPI::Init() - MPI::Finalize() cycles
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
        MPI::Init(argc, argv);
        LibGeoDecomp::Typemaps::initializeMaps();
        return true;
    }

    virtual bool tearDownWorld()
    {
        MPI::Finalize();
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
        TS_ASSERT_THROWS_NOTHING(  MPI::COMM_WORLD.Get_rank(); );
    }
};

static MPIGlobalFixture mpiSetupAndFinalizer;
#endif
