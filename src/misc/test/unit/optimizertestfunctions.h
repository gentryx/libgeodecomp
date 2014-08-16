// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/optimizer.h>
#include <cmath>
using namespace LibGeoDecomp;

namespace LibGeoDecomp {
class OptimizerTestFunctions{
public:
    class TestableEvaluator : public Optimizer::Evaluator
    {
    public:
        virtual ~TestableEvaluator()
        {}
        virtual double operator()(SimulationParameters params) = 0;

        TestableEvaluator():
            calls(0)
        {}

        double getGlobalMax()
        {
            return maxima[0];
        }

        std::vector<double> getLocalMax() {
            return maxima;
        }

        int getCalls() const
        {
            return calls;
        }

        void resetCalls()
        {
            calls = 0;
        }
    protected:
        int calls;
        std::vector<double> maxima;
    };

    class GoalFunction : public TestableEvaluator
    {
    public:
        GoalFunction():TestableEvaluator()
        {
            maxima.push_back(1000);
        }

        double operator()(SimulationParameters params)
        {
            int x = params["x"];
            int y = params["y"];
            ++calls;
            return 1000 - ((x - 5) * (x - 5)) - (y * y);
        }
    };

    class ThreeDimFunction : public TestableEvaluator
    {
    public:
        ThreeDimFunction()
        {
            maxima.push_back(2600);
        }

        double operator()(SimulationParameters params)
        {
            int x = params["x"];
            int y = params["y"];
            int z = params["z"];
            ++calls;
            return 1000 - ((z + x - 5) * (z - x - 5)) - (y * y);
        }
    };

    class FiveDimFunction : public TestableEvaluator
    {
    public:
        FiveDimFunction()
        {
            maxima.push_back(5000);
        }

        double operator()(SimulationParameters params)
        {
            ++calls;
            int v = params["v"];
            int w = params["w"];
            int x = params["x"];
            bool  by = params["y"];
            int y = 0;
            int z = params["z"];
                if (by) {
                    y = 1;
                }
            return
                5000 -
                ((v + 42) * ( v + 42) * 0.01) -
                (( 10 + w) * ( 10 + w)) -
                (x * x) -
                (1 - y) -
                ((z - 8) * (z - 8) * 0.1);
        }
    };

    class MultimodTwoDim : public TestableEvaluator
    {
    public:
        MultimodTwoDim()
        {
            maxima.push_back(4999.57);	//first value is global min
            maxima.push_back(4687.77);
        }

        double operator()(SimulationParameters params)
        {
            int x = params["x"];
            int y = params["y"];
            ++calls;
            return 4000 + (sin(x * 0.1) * 1000 * -1)+ y * y * -1;
        }
    };

    class JumpingFunction : public TestableEvaluator
    {
    public:
        JumpingFunction()
        {
            maxima.push_back(1103.9);
            maxima.push_back(1096);
        }

        double operator()(SimulationParameters params)
        {
            ++calls;
            int x = params["x"];
            int y = params["y"];

            if (x < 10 && x >= 10) {
                return 1100 + ((x * x * -0.1)- 4 + y * -0.1);
            } else {
                return 1100 + ((x * x * -0.1) + y * 0.1);
            }
        }
    };

    /**
     * for reference, please see http://en.wikipedia.org/wiki/Himmelblau%27s_function
     */
    class HimmelblauFunction : public TestableEvaluator
    {
    public:
        HimmelblauFunction()
        {
            maxima.push_back(1000);
        }

        double operator()(SimulationParameters params)
        {
            ++calls;
            int xi =  params["x"]; // range -5 - 5
            int yi =  params["y"];
            double x = (double) xi / (double) 100;
            double y = (double) yi / (double) 100;

            // (x^2 + y -11)^2 + (x + y^2 - 7)^2
            return 1000 - ( (((x * x) + y - (double) 11) * ( (x * x) + y - (double)11))
                    + (( x + (y * y) - (double) 7) * ( x + (y * y) - (double) 7)));
        }
    };
    
    class HimmelblauFunctionDouble : public TestableEvaluator
    {
    public:
        HimmelblauFunctionDouble() {
            maxima.push_back(1000);
        }

        double operator()(SimulationParameters params) {
            ++calls;
            double x = params["x"];
            double y = params["y"];
            //x = x / (double) 100;
            //y = y / (double) 100;
            // (x^2 + y -11)^2 + (x + y^2 - 7)^2
            return 1000 - ( (((x * x) + y - (double) 11) * ( (x * x) + y - (double)11))
                    + (( x + (y * y) - (double) 7) * ( x + (y * y) - (double) 7)));
        }
    };
    /**
     * for reference, please see http://en.wikipedia.org/wiki/Rosenbrock_function
     * tesfunction is valid for -1000 <= x <= 1000, -500 <= y 1500
     */
    class Rosenbrock2DFunction : public TestableEvaluator
    {
    public:
        Rosenbrock2DFunction()
        {
            maxima.push_back(3000);
        }
        double operator()(SimulationParameters params)
        {
            ++calls;
            int xi = params["x"];
            int yi = params["y"];
            double x = (double) xi / (double)500;
            double y = (double) yi / (double)500;
            return 3000 - (100 * (y - x * x) * (y -x * x)
                + (x - 1) * (x - 1));
        }
    };
    
    class Rosenbrock2DFunctionDouble : public TestableEvaluator
    {
    public:
        Rosenbrock2DFunctionDouble()
        {
            maxima.push_back(3000);
        }
        double operator()(SimulationParameters params)
        {
            ++calls;
            double x = params["x"];
            double y = params["y"];
            x = x/500;
            y = y/500;
            return 3000 - (100 * (y - x * x) * (y -x * x)
                + (x - 1) * (x - 1));
        }
    };

    class Rosenbrock5DFunction : public TestableEvaluator
    {
    public:
        Rosenbrock5DFunction()
        {
            maxima.push_back(100000);
        }
        double operator()(SimulationParameters params)
        {
            ++calls;
            int vi = params["v"];
            int wi = params["w"];
            int xi = params["x"];
            int yi = params["y"];
            int zi = params["z"];
            double v = (double) vi / (double) 500;
            double w = (double) wi / (double) 500;
            double x = (double) xi / (double) 500;
            double y = (double) yi / (double) 500;
            double z = (double) zi / (double) 500;
            return 100000 - ( 
                  (100 * (w - v * v) * (w - v * v) + (1 - v)*(1 - v))
                + (100 * (x - w * w) * (x - w * w) + (1 - w)*(1 - w))
                + (100 * (y - x * x) * (y - x * x) + (1 - x)*(1 - x))
                + (100 * (z - y * y) * (z - y * y) + (1 - z)*(1 - z)));
        }
    };

};
} // namespace LibGeoDecomp
