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
        params.addParameter("x", -5, 6);
        params.addParameter("y", -5, 6);
        PatternOptimizerTest::HimmelblauFunction eval;
        double test = eval(params);
        SimplexOptimizer optimizer(params);
        params =optimizer(10, eval);
        //TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.fitness);
    }

};

} // namespace LibGeoDecomp

