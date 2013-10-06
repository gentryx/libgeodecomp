#include <cxxtest/TestSuite.h>
#include <unistd.h>
#include <libgeodecomp/misc/chronometer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ChronometerTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        c = new Chronometer();
    }

    void tearDown()
    {
        delete c;
    }

    void testWorkRatio1()
    {
        {
            ScopedTimer t1 = c->tic(Chronometer::TOTAL_TIME);

            {
                ScopedTimer t2 = c->tic(Chronometer::COMPUTE_TIME);
                ScopedTimer::busyWait(30000);
            }

            ScopedTimer::busyWait(60000);
        }

        TS_ASSERT_DELTA(1.0/3.0, c->ratio(Chronometer::COMPUTE_TIME, Chronometer::TOTAL_TIME), 0.1);
    }

    void testWorkRatio2()
    {
        {
            ScopedTimer t1 = c->tic(Chronometer::TOTAL_TIME);
            {
                ScopedTimer t2 = c->tic(Chronometer::COMPUTE_TIME);
                ScopedTimer::busyWait(15000);
            }
            ScopedTimer::busyWait(30000);
            {
                ScopedTimer t2 = c->tic(Chronometer::COMPUTE_TIME);
                ScopedTimer::busyWait(15000);
            }
            ScopedTimer::busyWait(30000);
        }

        TS_ASSERT_DELTA(1.0/3.0, c->ratio(Chronometer::COMPUTE_TIME, Chronometer::TOTAL_TIME), 0.01);
    }

private:
    Chronometer *c;
};

}
