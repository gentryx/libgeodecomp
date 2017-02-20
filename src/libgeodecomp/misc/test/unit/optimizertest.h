// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/test/unit/optimizertestfunctions.h>
#include <libgeodecomp/io/logger.h>

//#define LIBGEODECOMP_DEBUG_LEVEL 4

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class GoalFunctionOptimizerTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::GoalFunction();
        params = SimulationParameters();
        params.addParameter("x",   0, 20);
        params.addParameter("y", -10, 10);
        LOG(Logger::INFO,"GoalFunction:")
    }
    void tearDown()
    {
        LOG(Logger::INFO,"Calls: " << eval.getCalls()
                 << std::endl
                 << "x: "<< (int) params["x"]
                 <<" y: " << (int) params["y"]
                 << std::endl)
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexChangedStart()
    {
        params["x"].setValue(17);
        params["y"].setValue(8);
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "SimplexOptimizertest with other start values: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testRightUpperCornerPatternDefault()
    {
        params["x"] = 20;
        params["y"] = 10;
        std::vector<double> stepwidth;
        std::vector<double> minStepwidth;
        for(std::size_t i = 0; i < params.size(); ++i){
            stepwidth.push_back(3);
            minStepwidth.push_back(1);
        }
        PatternOptimizer optimizer(params, stepwidth, minStepwidth);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, " testRightUpperCornerPatternDefault: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testRightUpperCornerSimplexDefault()
    {
        params["x"] = 20;
        params["y"] = 10;
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, " testRightUpperCornerSimplexDefault: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testRightLowerCornerPatternDefault()
    {
        params["x"] = 20;
        params["y"] = -10;
        std::vector<double> stepwidth;
        std::vector<double> minStepwidth;
        for(std::size_t i = 0; i < params.size(); ++i){
            stepwidth.push_back(3);
            minStepwidth.push_back(1);
        }
        PatternOptimizer optimizer(params, stepwidth, minStepwidth);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, " testRightLowerCornerPatternDefault: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testRightLowerCornerSimplexDefault()
    {
        params["x"] = 20;
        params["y"] = -10;
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, " testRightLowerCornerSimplexDefault: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }

private:
    SimulationParameters params;
    OptimizerTestFunctions::GoalFunction eval;
};

class ThreeDimFunctionOptimizerTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::ThreeDimFunction();
        params = SimulationParameters();
        params.addParameter("x", -40, 20);
        params.addParameter("y", -10, 10);
        params.addParameter("z", -10, 10);
        LOG(Logger::INFO,"ThreeDimFunction with dependend Parameters")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls()
                 << std::endl
                 << "x: "<< (int) params["x"]
                 << " y: " << (int) params["y"]
                 << " z: " << (int) params["z"]
                 << std::endl);
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::ThreeDimFunction eval;
};
class Multimodal2DTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::MultimodTwoDim();
        params = SimulationParameters();
        params.addParameter("x", -40, 40);
        params.addParameter("y", -50, 50);
        LOG(Logger::INFO, "Multimodal 2D Test:")
    }
     void tearDown()
     {
        LOG(Logger::INFO, "Calls: " << eval.getCalls()
                 << std::endl
                 << "x: "<< (int) params["x"]
                 <<" y: " << (int) params["y"]
                 << std::endl);
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.0001) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 25.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexModifiedParameters()
    {
        std::vector<double> s;
        for(std::size_t i = 0; i < params.size(); ++i) {
            s.push_back(2.0);
        }
        SimplexOptimizer optimizer(params, -17, 8.0, s);
        params = optimizer(5000,eval);
        TS_ASSERT(((eval.getGlobalMax() - 2.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with chaneged parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())

    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::MultimodTwoDim eval;
};
class DiscontinousFunctionTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::JumpingFunction();
        params = SimulationParameters();
        params.addParameter("x", -60, 60);
        params.addParameter("y", 0, 40);
        LOG(Logger::INFO, "Discontinous function test:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls()
                 << std::endl
                 << "x: "<< (int) params["x"]
                 <<" y: " << (int) params["y"]
                 << std::endl);
    }

    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }

private:
    SimulationParameters params;
    OptimizerTestFunctions::JumpingFunction eval;
};
class FiveDimensionsTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::FiveDimFunction();
        params = SimulationParameters();
        params.addParameter("v", -60, 60);
        params.addParameter("w", -20, 40);
        params.addParameter("x", -10, 10);
        std::vector<bool> set;
        set << false << true;
        params.addParameter("y", set);
        params.addParameter("z", -10, 40);
        LOG(Logger::INFO, "Five dimensions function:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "calls: " << eval.getCalls() << std::endl
                << "v:"   << (int) params["v"]
                << " w: " << (int) params["w"]
                << " x: " << (int) params["x"]
                << " y: " << (bool) params["y"]
                << " z: " << (int) params["z"]
                << std::endl);
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.getFitness());
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 20) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }

    void testSimplexOptimizedParameters()
    {
        std::vector<double> s;
        for(std::size_t i = 0; i < params.size(); ++i) {
            s.push_back(1.0);
        }
        SimplexOptimizer optimizer(params, 17.0 , 8.0, s);
        params = optimizer(20 ,eval);
        TS_ASSERT(((eval.getGlobalMax() - 10) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with optimized parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testHybridVersion()
    {
        SimplexOptimizer simplOptimizer(params);
        params = simplOptimizer(500 , eval);

        std::vector<double> s;
        for(std::size_t i = 0; i < params.size(); ++i) {
            s.push_back(6.0);
        }
        PatternOptimizer pattOptimizer(params,s);
        params = pattOptimizer(100, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), pattOptimizer.getFitness());
        LOG(Logger::INFO, "Simplex and Pattern Seach Hybrid: "
                        << std::endl << "getFitness(): " << pattOptimizer.getFitness())
    }

private:
    SimulationParameters params;
    OptimizerTestFunctions::FiveDimFunction eval;
};

class HimmelblauTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::HimmelblauFunction();
        params = SimulationParameters();
        params.addParameter("x", -500, 500);
        params.addParameter("y", -500, 500);
        LOG(Logger::INFO, "Himmelblau function:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls() << std::endl
                << "x: " << (int) params["x"]
                << " y: " << (int) params["y"]
                << std::endl);
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.01) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.2) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::HimmelblauFunction eval;

};

class HimmelblauDoubleTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::HimmelblauFunctionDouble();
        params = SimulationParameters();
        params.addParameter("x",(double)-5.0, (double)5.0, (double)0.00001);
        params.addParameter("y", (double)-5.0, (double)5.0, (double)0.00001);
        LOG(Logger::INFO, "HimmelblauFunctionDouble:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls() << std::endl
                << "x: " <<  (double) params["x"].getValue()
                << " y: " << (double) params["y"].getValue()
                << " granulatity from x " << params["x"].getGranularity()
                << std::endl);
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.01) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.01) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexChangedParameters()
    {
        SimplexOptimizer optimizer(params, 0.01, 4);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 0.2) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with optimized parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::HimmelblauFunctionDouble eval;

};

class Rosenbrock2DTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::Rosenbrock2DFunction();
        params = SimulationParameters();
        params.addParameter("x", -1000, 1000);
        params.addParameter("y", -500, 1500);
        LOG(Logger::INFO, "Rosenbrock-Function 2D:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls() << std::endl
                   << "x: " << (int) params["x"]
                   << " y: " << (int) params["y"]
                   << std::endl)
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 1.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 1.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::Rosenbrock2DFunction eval;
};
class Rosenbrock2DTestDouble : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::Rosenbrock2DFunctionDouble();
        params = SimulationParameters();
        params.addParameter("x", (double) -1000.0, (double)1000.0,0);
        params.addParameter("y", (double)-500.0, (double)1500.0,0);
        LOG(Logger::INFO, "Rosenbrock-Function 2D Double:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls() << std::endl
                   << "x: " << (double) params["x"]
                   << " y: " << (double) params["y"]
                   << std::endl)
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 1.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        SimplexOptimizer optimizer(params, 0.0000001,64);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 1.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::Rosenbrock2DFunctionDouble eval;
};
class Rosenbrock5DTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        eval = OptimizerTestFunctions::Rosenbrock5DFunction();
        params = SimulationParameters();
        params.addParameter("v", -1000, 1000);
        params.addParameter("w", -1000, 1000);
        params.addParameter("x", -1000, 1000);
        params.addParameter("y", -1000, 1000);
        params.addParameter("z", -1000, 1000);
        LOG(Logger::INFO, "Rosenbrock-Function 5D:")
    }
    void tearDown()
    {
        LOG(Logger::INFO, "Calls: " << eval.getCalls() << std::endl
                   << "v: " << (int) params["v"]
                   << " w: " << (int) params["W"]
                   << " x: " << (int) params["x"]
                   << " y: " << (int) params["y"]
                   << " z: " << (int) params["z"]
                   << std::endl)
    }
    void testPatternDefault()
    {
        PatternOptimizer optimizer(params);
        params = optimizer(5000, eval);
        TS_ASSERT(((eval.getGlobalMax() - 1.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "Patternoptimizertest with default parameters: "
                         << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexDefault()
    {
        // thats not the right way, wrong start parameters!
        SimplexOptimizer optimizer(params);
        params = optimizer(50, eval);
        TS_ASSERT(((eval.getGlobalMax() - 41000.0) < optimizer.getFitness()));
        LOG(Logger::INFO, "SimplexOptimizertest with default parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness())
    }
    void testSimplexOptimizedParameters()
    {
        params["v"].setValue(500);
        params["w"].setValue(500);
        params["x"].setValue(500);
        params["y"].setValue(500);
        params["z"].setValue(500);
        std::vector<double> s;
        for(std::size_t i = 0; i < params.size(); ++i) {
            s.push_back(4.0);
        }
        SimplexOptimizer optimizer(params, -0.1, 250.0, s);
        params = optimizer(20 ,eval);
        TS_ASSERT((eval.getGlobalMax() - 6.0 ) < optimizer.getFitness());
        LOG(Logger::INFO, "SimplexOptimizertest with optimized parameters: "
                        << std::endl << "getFitness(): " << optimizer.getFitness()
                        << "Global maximum: " <<  eval.getGlobalMax())
    }
private:
    SimulationParameters params;
    OptimizerTestFunctions::Rosenbrock5DFunction eval;
};
}
