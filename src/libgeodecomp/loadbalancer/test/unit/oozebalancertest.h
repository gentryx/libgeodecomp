#include <limits>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OozeBalancerTest1 : public CxxTest::TestSuite
{
public:
    void testConstructor()
    {
        // newLoadWeight gotta be in [0, 1]
        TS_ASSERT_THROWS(OozeBalancer(-1), std::invalid_argument&);
        TS_ASSERT_THROWS(OozeBalancer(1.1), std::invalid_argument&);
    }

    void checkExpectedOptimalDistribution(OozeBalancer::LoadVec expected, OozeBalancer::WeightVec loads, OozeBalancer::LoadVec relLoads)
    {
        OozeBalancer b(0.2);
        OozeBalancer::LoadVec actual = b.expectedOptimalDistribution(loads, relLoads);

        TS_ASSERT_EQUALS_DOUBLE_VEC(actual, expected);
    }

    void testExpectedOptimalDistribution1()
    {
        OozeBalancer::WeightVec loads(3);
        loads[0] = 4;
        loads[1] = 2;
        loads[2] = 5;

        OozeBalancer::LoadVec relLoads(3);
        relLoads[0] = 0.2;
        relLoads[1] = 0.9;
        relLoads[2] = 1.0;

        // the total load to distribute is 0.2 + 0.9 + 1.0 = 2.1, per
        // each node 0.7. The first node currently holds 4 items, his
        // load is 0.2, so each item costs 0.05 time. Similarly each
        // item for the second node takes 0.45 and for the third one
        // 0.2.
        OozeBalancer::LoadVec expected(3);
        // the fist node gets 10/9 items from the second one added...
        expected[0] = 4 + 10.0 / 9;
        // ...which gets itself another 1.5 items from the third one.
        expected[1] = 8.0 / 9 + 1.5;
        // the third one is now happy with just 3.5 items left.
        expected[2] = 3.5;

        checkExpectedOptimalDistribution(expected, loads, relLoads);
    }

    void testExpectedOptimalDistribution2()
    {
        OozeBalancer::WeightVec loads(3);
        loads[0] = 100;
        loads[1] = 0;
        loads[2] = 0;

        OozeBalancer::LoadVec relLoads(3);
        relLoads[0] = 0.8;
        relLoads[1] = 0.0;
        relLoads[2] = 0.0;

        OozeBalancer::LoadVec expected(3, 100.0 / 3);
        checkExpectedOptimalDistribution(expected, loads, relLoads);
    }

    void testOptDistWithElementsTooLargeForOneNode()
    {
        OozeBalancer::WeightVec loads(4);
        loads[0] = 2;
        loads[1] = 2;
        loads[2] = 3;
        loads[3] = 2;

        OozeBalancer::LoadVec relLoads(4);
        relLoads[0] = 0.13;
        relLoads[1] = 5.0;
        relLoads[2] = 0.21;
        relLoads[3] = 0.1;
        // target load per node:
        //   1/n * \sum relLoads == 1/4 * 5.46 == 1.34

        OozeBalancer::LoadVec expected(4);
        expected[0] =
            2 +        // both elements from loads[0]          => 0.13 load
            0.492;     // 49.2% of the 1st one in loads[1]     => 1.23 load
                       // -------------------------------------------------
                       //                                         1.36 load
        expected[1] =
            0.508 +    // remainder of the 1st one in loads[1] => 1.27 load
            0.036;     // 3.6% of the 2nd part of loads[1]     => 0.09 load
                       // -------------------------------------------------
                       //                                         1.36 load
        expected[2] =
            0.544;     // another 54.4% of loads[1]'s 2nd      => 1.36 load
                       // -------------------------------------------------
                       //                                         1.36 load
        expected[3] =
            0.42 +     // remainder of the 2nd one in loads[1] => 1.05 load
            3 +        // all three of loads[2]                => 0.21 load
            2;         // both of loads[3]                     => 0.10 load
                       // -------------------------------------------------
                       //                                         1.36 load

        checkExpectedOptimalDistribution(expected, loads, relLoads);
    }
};


class OozeBalancerTest2 : public CxxTest::TestSuite
{
public:

    void checkBoundaryConditions(OozeBalancer::WeightVec oldLoads, OozeBalancer::WeightVec newLoads)
    {
        TS_ASSERT_EQUALS(oldLoads.size(), newLoads.size());
        TS_ASSERT_EQUALS(sum(oldLoads), sum(newLoads));
    }

    /**
     * the time one node needs for calculation depends on its share of
     * items  loads, how complex each item is itemLoads and how
     * fast the node itself is  nodeSpeeds .
     */
    OozeBalancer::LoadVec calcRelLoads(OozeBalancer::WeightVec loads, OozeBalancer::LoadVec itemLoads, OozeBalancer::LoadVec nodeSpeeds)
    {
        OozeBalancer::LoadVec ret(nodeSpeeds.size(), 0);
        unsigned cursor = 0;
        for (unsigned node = 0; node < ret.size(); node++) {
            for (unsigned i = 0; i < loads[node]; i++)
                ret[node] += itemLoads[cursor++];
            ret[node] /= nodeSpeeds[node];
        }

        return ret;
    }

    void checkConvergence(OozeBalancer::WeightVec startLoads, OozeBalancer::LoadVec itemLoads, OozeBalancer::LoadVec nodeSpeeds)
    {
        OozeBalancer b;

        OozeBalancer::WeightVec oldLoads = startLoads;
        OozeBalancer::WeightVec newLoads;

        for (int i = 0; i < 64; i++) {
            OozeBalancer::LoadVec relLoads = calcRelLoads(oldLoads, itemLoads, nodeSpeeds);
            newLoads = b.balance(oldLoads, relLoads);
            checkBoundaryConditions(newLoads, oldLoads);
            oldLoads = newLoads;
        }

        double targetLoadPerNode = sum(itemLoads) / sum(nodeSpeeds);

        OozeBalancer::LoadVec relLoads = calcRelLoads(newLoads, itemLoads, nodeSpeeds);
        for (unsigned i = 0; i < relLoads.size(); i++) {
            // 5% is a good approximation after so few steps
            TS_ASSERT_DELTA(targetLoadPerNode, relLoads[i], targetLoadPerNode * 0.05);
        }
    }

    OozeBalancer::LoadVec itemLoads1()
    {
        OozeBalancer::LoadVec itemLoads(1500);
        for (int i = 0; i < 400; i++)
            itemLoads[i] = 0.1;
        for (int i = 400; i < 1300; i++)
            itemLoads[i] = 1.0;
        for (int i = 1300; i < 1500; i++)
            itemLoads[i] = 0.3;
        return itemLoads;
    }

    OozeBalancer::LoadVec nodeSpeeds1()
    {
        OozeBalancer::LoadVec nodeSpeeds(5);
        nodeSpeeds[0] = 20;
        nodeSpeeds[1] = 15;
        nodeSpeeds[2] = 25;
        nodeSpeeds[3] = 10;
        nodeSpeeds[4] = 55;
        return nodeSpeeds;
    }

    void testConvergence1()
    {
        // all items to node 0
        OozeBalancer::WeightVec startLoads(5, 0);
        startLoads[0] = 1500;

        checkConvergence(startLoads, itemLoads1(), nodeSpeeds1());
    }

    void testConvergence2()
    {
        // equidistribution
        OozeBalancer::WeightVec startLoads(5, 300);

        checkConvergence(startLoads, itemLoads1(), nodeSpeeds1());
    }

    OozeBalancer::LoadVec tLoads()
    {
        OozeBalancer::LoadVec dLoads(5);
        dLoads[0] = 1.6;
        dLoads[1] = 5.7;
        dLoads[2] = 9.4;
        dLoads[3] = 3.4;
        dLoads[4] = 6.9;
        return dLoads;
    }

    void testEqualize1()
    {
        OozeBalancer::WeightVec expected(5);
        expected[0] = 1;
        expected[1] = 6;
        expected[2] = 9;
        // equalize() is expected to accumulate rounded off (and up)
        // fractions and honor this balance when rounding. so when
        // rounding the 3.4 at position 3 it should take into account
        // the current balance of 0.7 (0.6 - 0.3 + 0.4) and use this
        // to round up.
        expected[3] = 4;
        expected[4] = 7;

        TS_ASSERT_EQUALS(expected, OozeBalancer().equalize(tLoads()));
    }

    void testEqualize2()
    {
        OozeBalancer::LoadVec loads(1, 12.1);
        OozeBalancer::WeightVec expected(1, 12);
        TS_ASSERT_EQUALS(expected, OozeBalancer().equalize(loads));
    }

    void testEqualize3()
    {
        OozeBalancer::LoadVec loads(1, 11.9);
        OozeBalancer::WeightVec expected(1, 12);
        TS_ASSERT_EQUALS(expected, OozeBalancer().equalize(loads));
    }

    void testLinearCombo()
    {
        OozeBalancer::WeightVec base(5);
        base[0] = 4;
        base[1] = 8;
        base[2] = 2;
        base[3] = 5;
        base[4] = 7;

        OozeBalancer::LoadVec expected(5);
        expected[0] = 4 * 0.9 + 1.6 * 0.1;
        expected[1] = 8 * 0.9 + 5.7 * 0.1;
        expected[2] = 2 * 0.9 + 9.4 * 0.1;
        expected[3] = 5 * 0.9 + 3.4 * 0.1;
        expected[4] = 7 * 0.9 + 6.9 * 0.1;

        TS_ASSERT_EQUALS_DOUBLE_VEC(
            expected,
            OozeBalancer(0.1).linearCombo(base, tLoads()));
    }

    void testZeroRelativeLoads()
    {
        OozeBalancer::WeightVec loads(5, 100);
        TS_ASSERT_EQUALS(loads, OozeBalancer().balance(loads, OozeBalancer::LoadVec(5, 0)));
    }
};

}
