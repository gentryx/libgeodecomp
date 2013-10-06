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
                ScopedTimer::busyWait(100000);
            }

            ScopedTimer::busyWait(200000);
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

        TS_ASSERT_DELTA(1.0/3.0, c->ratio(Chronometer::COMPUTE_TIME, Chronometer::TOTAL_TIME), 0.1);
    }

    void testAddition()
    {
        Chronometer c1;
        Chronometer c2;
        Chronometer c3;

        {
            ScopedTimer t1 = c1.tic(Chronometer::PATCH_ACCEPTERS_TIME);
            ScopedTimer t2 = c2.tic(Chronometer::PATCH_PROVIDERS_TIME);

            {
                ScopedTimer t = c1.tic(Chronometer::COMPUTE_TIME);
                ScopedTimer::busyWait(15000);
            }
            {
                ScopedTimer t = c2.tic(Chronometer::COMPUTE_TIME);
                ScopedTimer::busyWait(15000);
            }
        }

        c3 = c1 + c2;
        TS_ASSERT_EQUALS(c3.interval(Chronometer::COMPUTE_TIME),
                         c1.interval(Chronometer::COMPUTE_TIME) +
                         c2.interval(Chronometer::COMPUTE_TIME));

        TS_ASSERT_EQUALS(c3.interval(Chronometer::PATCH_ACCEPTERS_TIME),
                         c1.interval(Chronometer::PATCH_ACCEPTERS_TIME));

        TS_ASSERT_EQUALS(c3.interval(Chronometer::PATCH_PROVIDERS_TIME),
                         c2.interval(Chronometer::PATCH_PROVIDERS_TIME));
    }

private:
    Chronometer *c;
};

}
