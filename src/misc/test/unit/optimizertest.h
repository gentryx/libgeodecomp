// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/io/logger.h>

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

        LOG(Logger::INFO,"Optimizer is running! ");

        Optimizer optimizer(params);
        GoalFunction eval;
        optimizer(5000, eval);

        TS_ASSERT_EQUALS(1000, optimizer.fitness);
    }
};
}
