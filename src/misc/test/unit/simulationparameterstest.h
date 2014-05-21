#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimulationFactoryTest : public CxxTest::TestSuite
{
public:

    // void testUsage()
    // {
    //     SimulationParameters params;
    //     params.addParameter("x",   0, 20);
    //     params.addParameter("y", -10, 10);

    //     std::cout << params["x"] << "\n";
    //     std::cout << params["x"].min << "\n";
    //     std::cout << params["x"].max << "\n";

    //     std::cout << params["y"] << "\n";
    //     std::cout << params["y"].min << "\n";
    //     std::cout << params["y"].max << "\n";

    //     params["x"].getIndex();
    //     params["x"] += 5;
    // }

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
};

}
