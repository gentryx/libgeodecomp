// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/test/unit/optimizertestfunctions.h>
#include <libgeodecomp/io/logger.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {


class PatternOptimizerTest : public CxxTest::TestSuite
{
public:



    void testBasic()
    {
        LOG(Logger::INFO,"PatternOptimizer is running!"<< std::endl);

        SimulationParameters params;
        params.addParameter("x",   0, 20);
        params.addParameter("y", -10, 10);
        PatternOptimizer optimizer(params);
        OptimizerTestFunctions::GoalFunction eval;
        params = optimizer(5000, eval);
        TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.fitness);
        LOG(Logger::INFO, "Test 1: "
                 << std::endl << "fitness: " << optimizer.fitness
                 <<  " calls: " << eval.getCalls()
                 << std::endl
                 << "x: "<< (int)params["x"]
                 <<" y: " << (int)params["y"]
                 << std::endl);
    }

    void test3DwithDependetParameters()
    {
        SimulationParameters params2;
        params2.addParameter("x", -40, 20);
        params2.addParameter("y", -10, 10);
        params2.addParameter("z", -10, 10);
        PatternOptimizer optimizer2(params2);
        OptimizerTestFunctions::ThreeDimFunction eval2;
        params2 = optimizer2(100,eval2);
        TS_ASSERT_EQUALS(eval2.getGlobalMax(), optimizer2.fitness);


        LOG(Logger::INFO, "Test 2, dependend parameters:"
                 << std::endl << "fitness: " << optimizer2.fitness
                 <<  " calls: " << eval2.getCalls()
                 << std::endl
                 << "x: "<< (int)params2["x"]
                 << " y: " << (int)params2["y"]
                 << " z: " << (int)params2["z"]
                 << std::endl);
    }

    void testMultimodal2D()
    {
        SimulationParameters params3;
        params3.addParameter("x", -40, 40);
        params3.addParameter("y", -50, 50);
        PatternOptimizer optimizer3(params3);
        OptimizerTestFunctions::MultimodTwoDim eval3;
        params3 = optimizer3(100,eval3);
        TS_ASSERT(((eval3.getGlobalMax() - 0,0001) < optimizer3.fitness));
        LOG(Logger::INFO,  "Test 3 multimodal function: " << std::endl
                 << "fitness: " << optimizer3.fitness
                 <<  " calls: " << eval3.getCalls()
                 << std::endl
                 << "x: "<< (int)params3["x"]
                 <<" y: " << (int)params3["y"]
                 << std::endl);
    }

    void testDiscontinousFunction()
    {
        SimulationParameters params4;
        params4.addParameter("x", -60, 60);
        params4.addParameter("y", 0, 40);
        PatternOptimizer optimizer4(params4);
        OptimizerTestFunctions::JumpingFunction eval4;
        params4 = optimizer4(100,eval4);
        TS_ASSERT_EQUALS(eval4.getGlobalMax(), optimizer4.fitness);
        LOG(Logger::INFO,  "Test 4 discontinuous function:"
                 << std::endl << "fitness: "
                 << optimizer4.fitness<<  " calls: " << eval4.getCalls()
                 << std::endl
                 << "x: "<< (int)params4["x"]
                 <<" y: " << (int)params4["y"]
                 << std::endl);
    }

    void test5Dimensions()
    {
        SimulationParameters params5;
        params5.addParameter("v", -60, 60);
        params5.addParameter("w", -20, 40);
        params5.addParameter("x", -10, 10);
        params5.addParameter("y", 0, 2);
        params5.addParameter("z", -10, 40);
        PatternOptimizer optimizer5(params5);
        OptimizerTestFunctions::FiveDimFunction eval5;
        params5 = optimizer5(100,eval5);
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

    void testHimmelblau()
    {
        SimulationParameters params6;
        params6.addParameter("x", -500, 500);
        params6.addParameter("y", -500, 500);

        PatternOptimizer optimizer6(params6);
        OptimizerTestFunctions::HimmelblauFunction eval6;
        params6 = optimizer6(100, eval6);
        TS_ASSERT(((eval6.getGlobalMax() - 0,0001) < optimizer6.fitness));

        LOG(Logger::INFO, "Test 6, Himmelblau:" << std::endl
                << "fitness: " << optimizer6.fitness
                << " calls: " << eval6.getCalls() << std::endl
                << " x: " << (int)params6["x"]
                << " y: " << (int)params6["y"]
                << std::endl);
    }
};
}
