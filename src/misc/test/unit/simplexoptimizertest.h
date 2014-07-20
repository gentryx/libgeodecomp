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
//        params["x"].setValue(0);
//        params["y"].setValue(5);
        PatternOptimizerTest::HimmelblauFunction eval;
        double test = eval(params);
        SimplexOptimizer optimizer(params);
        params =optimizer(20, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.fitness);
        std::cout << "x: " << (int) params["x"] << std::endl << "y: " << (int) params["y"] << "Calls: " << eval.getCalls() <<  std::endl;
    

        SimulationParameters params5;
        params5.addParameter("v", -60, 60);
        params5.addParameter("w", -20, 40);
        params5.addParameter("x", -10, 10);
        params5.addParameter("y", 0, 2);
        params5.addParameter("z", -10, 40);
        SimplexOptimizer optimizer5(params5);
        PatternOptimizerTest::FiveDimFunction eval5;
        params5 = optimizer5(40,eval5);
        TS_ASSERT_EQUALS(eval5.getGlobalMax(), optimizer5.fitness);
        LOG(Logger::INFO,  "Test 5, five dimensions: "
                << std::endl<< "fitness: " << optimizer5.fitness
                <<  " calls: " << eval5.getCalls() << std::endl
                << "v:"   << (int)params5["v"]
                << " w: " << (int)params5["w"]
                << " x: " << (int)params5["x"]
                << " y: " << (int)params5["y"]
                << " z: " << (int)params5["z"]
                << std::endl);
    }
};
} // namespace LibGeoDecomp

