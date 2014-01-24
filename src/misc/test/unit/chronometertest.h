#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif

#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/stringops.h>

#include <cxxtest/TestSuite.h>

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
            TimeTotal t(c);
            {
                TimeCompute t(c);
                ScopedTimer::busyWait(100000);
            }

            ScopedTimer::busyWait(200000);
        }

        double ratio = c->ratio<TimeCompute, TimeTotal>();
        TS_ASSERT_DELTA(1.0/3.0, ratio, 0.1);
    }

    void testWorkRatio2()
    {
        {
            TimeTotal t(c);
            {
                TimeComputeGhost t(c);
                ScopedTimer::busyWait(100000);
            }

            ScopedTimer::busyWait(300000);
            {
                TimeComputeInner t(c);
                ScopedTimer::busyWait(200000);
            }
            ScopedTimer::busyWait(300000);
        }

        double ratio;

        ratio = c->ratio<TimeCompute, TimeTotal>();
        TS_ASSERT_DELTA(1.0/3.0, ratio, 0.05);

        ratio = c->ratio<TimeComputeGhost, TimeTotal>();
        TS_ASSERT_DELTA(1.0/9.0, ratio, 0.05);

        ratio = c->ratio<TimeComputeInner, TimeTotal>();
        TS_ASSERT_DELTA(2.0/9.0, ratio, 0.05);
    }

    void testAddition()
    {
        Chronometer c1;
        Chronometer c2;
        Chronometer c3;

        {
            TimePatchAccepters t1(&c1);
            TimePatchProviders t2(&c2);

            {
                TimeCompute t(&c1);
                ScopedTimer::busyWait(15000);
            }
            {
                TimeCompute t(&c2);
                ScopedTimer::busyWait(15000);
            }
        }

        c3 = c1 + c2;
        TS_ASSERT_EQUALS(c3.interval<TimeCompute>(),
                         c1.interval<TimeCompute>() +
                         c2.interval<TimeCompute>());

        TS_ASSERT_EQUALS(c3.interval<TimePatchAccepters>(),
                         c1.interval<TimePatchAccepters>());

        TS_ASSERT_EQUALS(c3.interval<TimePatchProviders>(),
                         c2.interval<TimePatchProviders>());
    }

    void testReport()
    {
        {
            TimeTotal t(c);
            {
                TimeCompute t(c);
                ScopedTimer::busyWait(15000);
            }

            ScopedTimer::busyWait(30000);
            {
                TimeCompute t(c);
                ScopedTimer::busyWait(15000);
            }
            ScopedTimer::busyWait(30000);
        }

        std::vector<std::string> lines = StringOps::tokenize(c->report(), "\n");
        TS_ASSERT_EQUALS(lines.size()        , Chronometer::NUM_INTERVALS);
        TS_ASSERT_LESS_THAN_EQUALS(std::size_t(6),  Chronometer::NUM_INTERVALS);

        for (std::size_t i = 0; i < Chronometer::NUM_INTERVALS; ++i) {
            std::vector<std::string> tokens = StringOps::tokenize(lines[i], ":");
            TS_ASSERT_EQUALS(std::size_t(2), tokens.size());
            TS_ASSERT_LESS_THAN_EQUALS(0, StringOps::atof(tokens[1]));
        }
    }

    void testAddTime()
    {
        TS_ASSERT_EQUALS(0, c->interval<TimeCompute>());
        c->addTime<TimeCompute>(5);
        TS_ASSERT_EQUALS(5, c->interval<TimeCompute>());
        c->addTime<TimeCompute>(4);
        TS_ASSERT_EQUALS(9, c->interval<TimeCompute>());
    }

    void testTock()
    {
        double t = ScopedTimer::time();
        ScopedTimer::busyWait(15000);
        c->tock<TimeComputeInner>(t);
        TS_ASSERT_LESS_THAN_EQUALS(0.015, c->interval<TimeComputeInner>());
    }

private:
    Chronometer *c;
};

}
