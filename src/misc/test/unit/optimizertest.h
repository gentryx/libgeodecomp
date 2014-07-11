#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/patternoptimizer.h>


// Mathias is testing
#define MATHIAS_DBG


using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OptimizerTest : public CxxTest::TestSuite
{
public:
    class GoalFunction : public Optimizer::Evaluator
    {
    public:
        double operator()(SimulationParameters params)
        {
            int x = params["x"];
            int y = params["y"];

            return 1000 - ((x - 5) * (x - 5)) - (y * y);
        }
    };

    void testBasic()
    {
        SimulationParameters params;
        params.addParameter("x",   0, 20);
        params.addParameter("y", -10, 10);
#ifdef MATHIAS_DBG
std::cout<< std::endl <<"Optimizer is running! "<< std::endl;
#endif //MATHIAS_DBG
        Optimizer optimizer(params);
        GoalFunction eval;
        optimizer(5000, eval);

        TS_ASSERT_EQUALS(1000, optimizer.fitness);
    }

};

class PatternOptimizerTest : public CxxTest::TestSuite
{
public:
    class GoalFunction : public Optimizer::Evaluator
    {
    public:
        double operator()(SimulationParameters params)
        {
            int x = params["x"];
            int y = params["y"];

            return 1000 - ((x - 5) * (x - 5)) - (y * y);
        }
    };

    void testBasic()
    {
        SimulationParameters params;
        params.addParameter("x",   0, 20);
        params.addParameter("y", -10, 10);
#define MATHIAS_DBG
std::cout<< std::endl <<"PatternOptimizer is running!"<< std::endl;
#endif //MATHIAS_DBG
        PatternOptimizer optimizer(params);
        GoalFunction eval;
        optimizer(5000, eval);

        TS_ASSERT_EQUALS(1000, optimizer.fitness);
    }

};

}
