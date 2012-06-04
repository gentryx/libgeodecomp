#include <cxxtest/TestSuite.h>
#include <unistd.h>
#include <libgeodecomp/misc/chronometer.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class ChronometerTest : public CxxTest::TestSuite
{
private:
    Chronometer *c;

public:
    void setUp() 
    {
        c = new Chronometer();
    }

    
    void tearDown()
    {
        delete c;
    }


    void testTimeConsecutive()
    {
        for (int i = 0; i < 1024; i++) {
            TS_ASSERT(c->time() <= c->time());
        }
    }


    void testTimeOverflow()
    {
        long long tOld = c->time();
        long long tNew;
        for (int i = 0; i < 4; i++) {
            sleep(1);
            tNew = c->time();
            TS_ASSERT(tNew > tOld);
            tOld = tNew;
        }
    }


    void testMustCallTic()
    {
        TS_ASSERT_THROWS(c->toc(), std::logic_error);
    }


    void testWorkRatio1()
    {
        c->_cycleStart -= 1; // avoidCyclelength of 0 due to low resolution
        TS_ASSERT_EQUALS(0, c->nextCycle());
    }


    void testWorkRatio2()
    {
        c->tic();
        usleep(300000);
        c->toc();
        usleep(600000);
        TS_ASSERT_DELTA(1.0/3.0, c->nextCycle(), 0.2);
    }


    void testWorkRatio3()
    {
        c->tic();
        usleep(150000);
        c->toc();
        usleep(300000);
        c->tic();
        usleep(150000);
        c->toc();
        usleep(300000);
        TS_ASSERT_DELTA(1.0/3.0, c->nextCycle(), 0.2);
    }
};

};
