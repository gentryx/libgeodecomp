// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/test/unit/patternoptimizertest.h>
#include <libgeodecomp/io/logger.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimplexOptimizerTest :  public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        SimulationParameters params;
        params.addParameter("x", -40, 20);
        params.addParameter("y", -10, 10);
        params.addParameter("z", -10,10);
        PatternOptimizerTest::ThreeDimFunction eval;
        double test = eval(params);
        SimplexOptimizer optimizer(params);
        optimizer(10,eval);
        TS_ASSERT_EQUALS(1000, optimizer.fitness);
    }

};

} // namespace LibGeoDecomp

