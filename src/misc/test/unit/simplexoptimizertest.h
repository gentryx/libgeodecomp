// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/test/unit/patternoptimizertest.h>
#include <libgeodecomp/io/logger.h>

//#define LIBGEODECOMP_DEBUG_LEVEL 4

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimplexOptimizerTest :  public CxxTest::TestSuite
{
public:
    void testHimmelblau()
    {
        LOG(Logger::INFO,"SimplexOptimizer is running!"<< std::endl);
        // Test 1
        {
            SimulationParameters params;
            params.addParameter("x",   0, 20);
            params.addParameter("y", -10, 10);
            //SimplexOptimizer optimizer(params);
            std::vector<double> s;
            for(std::size_t i = 0; i < params.size(); ++i) {
                s.push_back(1.0);
            }
            SimplexOptimizer optimizer(params, s, 4.0, -1.0);
            PatternOptimizerTest::GoalFunction eval;
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
        {
            // Test2 3d, dependent parameters
            SimulationParameters params;
            params.addParameter("x", -40, 20);
            params.addParameter("y", -10, 10);
            params.addParameter("z", -10, 10);
            SimplexOptimizer optimizer(params);
            PatternOptimizerTest::ThreeDimFunction eval;
            params = optimizer(100,eval);
            TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.fitness);
            LOG(Logger::INFO, "Test 2, dependend parameters:"
                     << std::endl << "fitness: " << optimizer.fitness
                     <<  " calls: " << eval.getCalls()
                     << std::endl
                     << "x: "<< (int)params["x"]
                     << " y: " << (int)params["y"]
                     << " z: " << (int)params["z"]
                     << std::endl);
        }
        {
            // Test3 Multimodal 2d
            SimulationParameters params;
            params.addParameter("x", -40, 40);
            params.addParameter("y", -50, 50);
            std::vector<double> s;
            for(std::size_t i = 0; i < params.size(); ++i) {
                s.push_back(1.0);
            }
            SimplexOptimizer optimizer(params,s , 16, -1);
            PatternOptimizerTest::MultimodTwoDim eval;
            params = optimizer(100,eval);
            TS_ASSERT(((eval.getGlobalMax() - 0,0001) < optimizer.fitness));
            LOG(Logger::INFO,  "Test 3 multimodal function: "<< std::endl
                     << "fitness: " << optimizer.fitness
                     <<  " calls: " << eval.getCalls()
                     << std::endl
                     << "x: "<< (int)params["x"]
                     <<" y: " << (int)params["y"]
                     << std::endl);
        }
        {
            //Test4 discontinuous function:
            SimulationParameters params;
            params.addParameter("x", -60, 60);
            params.addParameter("y", 0, 40);
            SimplexOptimizer optimizer(params);
            PatternOptimizerTest::JumpingFunction eval;
            params = optimizer(100,eval);
            TS_ASSERT_EQUALS(eval.getGlobalMax(), optimizer.fitness);
            LOG(Logger::INFO,  "Test 4 discontinuous function:"
                     << std::endl << "fitness: "
                     << optimizer.fitness<<  " calls: " << eval.getCalls()
                     << std::endl
                     << "x: "<< (int)params["x"]
                     <<" y: " << (int)params["y"]
                     << std::endl);
        }
        {
            SimulationParameters params5;
            params5.addParameter("v", -60, 60);
            params5.addParameter("w", -20, 40);
            params5.addParameter("x", -10, 10);
            params5.addParameter("y", 0, 2);
            params5.addParameter("z", -10, 40);
            SimplexOptimizer optimizer5(params5);
            PatternOptimizerTest::FiveDimFunction eval5;
            params5 = optimizer5(20 ,eval5);
            TS_ASSERT_EQUALS(eval5.getGlobalMax(), optimizer5.fitness);
            LOG(Logger::INFO,  "Test 5, five dimensions with default Parameters: "
                    << std::endl<< "fitness: " << optimizer5.fitness
                    <<  " calls: " << eval5.getCalls() << std::endl
                    << "v:"   << (int)params5["v"]
                    << " w: " << (int)params5["w"]
                    << " x: " << (int)params5["x"]
                    << " y: " << (int)params5["y"]
                    << " z: " << (int)params5["z"]
                    << std::endl);
        }

        {
            SimulationParameters params5;
            params5.addParameter("v", -60, 60);
            params5.addParameter("w", -20, 40);
            params5.addParameter("x", -10, 10);
            params5.addParameter("y", 0, 2);
            params5.addParameter("z", -10, 40);
            std::vector<double> s;
            for(std::size_t i = 0; i < params5.size(); ++i) {
                s.push_back(1.0);
            }
            SimplexOptimizer optimizer5(params5, s, 8.0, 17.0);
          // SimplexOptimizer optimizer5(params5);
            PatternOptimizerTest::FiveDimFunction eval5;
            params5 = optimizer5(20 ,eval5);
            TS_ASSERT_EQUALS(eval5.getGlobalMax(), optimizer5.fitness);
            LOG(Logger::INFO,  "Test 5, five dimensions with individual Parameters: "
                    << std::endl<< "fitness: " << optimizer5.fitness
                    <<  " calls: " << eval5.getCalls() << std::endl
                    << "v:"   << (int)params5["v"]
                    << " w: " << (int)params5["w"]
                    << " x: " << (int)params5["x"]
                    << " y: " << (int)params5["y"]
                    << " z: " << (int)params5["z"]
                    << std::endl);
        }

        {
            SimulationParameters params;
            params.addParameter("x", -500, 500);
            params.addParameter("y", -500, 500);
            params["x"].setValue(20);
            params["y"].setValue(20);
            PatternOptimizerTest::HimmelblauFunction eval;
            double test = eval(params);
            std::vector<double> s;
            for(std::size_t i = 0; i < params.size(); ++i) {
                s.push_back(1.0);
            }
            SimplexOptimizer optimizer(params, s, 4.0, -1.0);
            params =optimizer(40, eval);
            TS_ASSERT(((eval.getGlobalMax() -0,0001) < optimizer.fitness));
            LOG(Logger::INFO, "Test Himmelblau with simplexOptimizer" << std::endl
                    << std::endl << "fitness: " << optimizer.fitness << std::endl
                    << "Calls: " << eval.getCalls() <<  std::endl
                    << "x: " << (int) params["x"]
                    << "y: " << (int) params["y"] << std::endl)
        }

        {
            SimulationParameters params;
            params.addParameter("x", -500, 500);
            params.addParameter("y", -500, 500);
            params["x"].setValue(20);
            params["y"].setValue(20);
            PatternOptimizerTest::HimmelblauFunction eval;
            double test = eval(params);
            SimplexOptimizer optimizer(params);
            params =optimizer(40, eval);
            TS_ASSERT((eval.getGlobalMax() -0,0001 < optimizer.fitness));
            LOG(Logger::INFO, "Test Himmelblau with simplexOptimizer and default Values:"
                    << std::endl << "fitness: " << optimizer.fitness << std::endl
                    << "Calls: " << eval.getCalls() <<  std::endl
                    << "x: " << (int) params["x"]
                    << "y: " << (int) params["y"] << std::endl)
        }
    }
};
} // namespace LibGeoDecomp
