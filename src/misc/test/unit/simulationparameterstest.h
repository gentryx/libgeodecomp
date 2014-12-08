#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/stringops.h>
#include <cuda.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimulationFactoryTest : public CxxTest::TestSuite
{
public:
    double eval(SimulationParameters params)
    {
        int x = params["x"];
        int y = params["y"];

        return 100 - ((x - 5) * (x - 5)) - (y * y);
    }

    void testOptimizableParameterInterface1()
    {
        std::vector<int> values;
        values << -10
               << -5
               << 0
               << 5
               << 10;
        SimulationParameters params;
        params.addParameter("x", 0, 20);
        params.addParameter("y", values);

        for (int t = 0; t < 10000; ++t) {
            SimulationParameters newParams = params;
            for (std::size_t i = 0; i < params.size(); ++i) {
                newParams[i] += ((rand() % 11) - 5) * newParams[i].getGranularity();
            }

            if (eval(newParams) > eval(params)) {
                params = newParams;
            }
        }

        TS_ASSERT_EQUALS(100, eval(params));
    }

    void testOptimizableParameterInterface2()
    {
        std::vector<int> values;
        values << -10
               << -5
               << 0
               << 5
               << 10;
        SimulationParameters params;
        params.addParameter("x", 0, 20);
        params.addParameter("y", values);

        double x = params[0].getValue();
        x += params[0].getGranularity() * 5;

        double oldFitness = eval(params);
        params[0].setValue(x);
        x = params[0].getValue();
        double newFitness = eval(params);

        TS_ASSERT_EQUALS(newFitness - oldFitness, 25);
    }

    void testBasic()
    {
        SimulationParameters params;
        params.addParameter("foo", 1, 5);

        std::vector<std::string> set1;
        std::vector<bool> set2;
        std::vector<int> set3;
        std::vector<double> set4;

        set1 << "United we stand"
             << "Together we can't fall"
             << "Now and forever"
             << "I found where I belong";

        set2 << false << true;

        set3 << 9
             << 3
             << 7
             << 1;

        set4 << 3.1
             << 1.3
             << 1.5
             << 2.0
             << 2.1
             << 2.2
             << 3.3
             << 3.4
             << 3.5;

        params.addParameter("bar1", set1);
        params.addParameter("bar2", set2);
        params.addParameter("bar3", set3);
        params.addParameter("bar4", set4);

        TS_ASSERT(params["bar1"] != "Now and Forever");
        TS_ASSERT(params["bar1"] == "United we stand");
        TS_ASSERT(params["bar1"] != true);
        TS_ASSERT(params["bar1"] != false);
        TS_ASSERT(params["bar1"] != 3);
        TS_ASSERT(params["bar1"] != 9);
        TS_ASSERT(params["bar1"] != 2.2);
        TS_ASSERT(params["bar1"] != 3.1);

        TS_ASSERT(params["bar2"] != "Now and Forever");
        TS_ASSERT(params["bar2"] != "United we stand");
        TS_ASSERT(params["bar2"] != true);
        TS_ASSERT(params["bar2"] == false);
        TS_ASSERT(params["bar2"] != 3);
        TS_ASSERT(params["bar2"] != 9);
        TS_ASSERT(params["bar2"] != 2.2);
        TS_ASSERT(params["bar2"] != 3.1);

        TS_ASSERT(params["bar3"] != "Now and Forever");
        TS_ASSERT(params["bar3"] != "United we stand");
        TS_ASSERT(params["bar3"] != true);
        TS_ASSERT(params["bar3"] != false);
        TS_ASSERT(params["bar3"] != 3);
        TS_ASSERT(params["bar3"] == 9);
        TS_ASSERT(params["bar3"] != 2.2);
        TS_ASSERT(params["bar3"] != 3.1);

        TS_ASSERT(params["bar4"] != "Now and Forever");
        TS_ASSERT(params["bar4"] != "United we stand");
        TS_ASSERT(params["bar4"] != true);
        TS_ASSERT(params["bar4"] != false);
        TS_ASSERT(params["bar4"] != 3);
        TS_ASSERT(params["bar4"] != 9);
        TS_ASSERT(params["bar4"] != 2.2);
        TS_ASSERT(params["bar4"] == 3.1);

        params["bar1"] = "Now and Forever";
        params["bar2"] = true;
        params["bar3"] = 3;
        params["bar4"] = 2.2;

        TS_ASSERT(params["bar1"] == "Now and Forever");
        TS_ASSERT(params["bar1"] != true);
        TS_ASSERT(params["bar1"] != 3);
        TS_ASSERT(params["bar1"] != 2.2);

        TS_ASSERT(params["bar2"] != "Now and Forever");
        TS_ASSERT(params["bar2"] == true);
        TS_ASSERT(params["bar2"] != 3);
        TS_ASSERT(params["bar2"] != 2.2);

        TS_ASSERT(params["bar3"] != "Now and Forever");
        TS_ASSERT(params["bar3"] != true);
        TS_ASSERT(params["bar3"] == 3);
        TS_ASSERT(params["bar3"] != 2.2);

        TS_ASSERT(params["bar4"] != "Now and Forever");
        TS_ASSERT(params["bar4"] != true);
        TS_ASSERT(params["bar4"] != 3);
        TS_ASSERT(params["bar4"] == 2.2);

    }

    void testReplaceParameterAndToString()
    {
        SimulationParameters params;
        params.addParameter("foo", 1, 5);

        TS_ASSERT_EQUALS("Interval([1, 5], 0)", params["foo"].toString());

        params.replaceParameter("foo", 6, 9);
        int val = params["foo"];
        TS_ASSERT_EQUALS("Interval([6, 9], 0)", params["foo"].toString());

        std::vector<char> values;
        values << 'a'
               << 'b'
               << 'c';
        params.replaceParameter("foo", values);
        TS_ASSERT_EQUALS("DiscreteSet([a, b, c], 0)", params["foo"].toString());

        params["foo"] += 2;
        TS_ASSERT_EQUALS("DiscreteSet([a, b, c], 2)", params["foo"].toString());
    }

    void testToString()
    {
        std::stringstream buf;

        SimulationParameters params;
        params.addParameter("foo", 1, 5);
        params.addParameter("bar", 2, 4);
        buf << params;

        std::string expected = "SimulationParameters(\n  bar => Interval([2, 4], 0)\n  foo => Interval([1, 5], 0)\n)\n";
        TS_ASSERT_EQUALS(expected, buf.str());
    }
};

}
