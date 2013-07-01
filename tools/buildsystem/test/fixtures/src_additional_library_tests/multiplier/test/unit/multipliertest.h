#include <cxxtest/TestSuite.h>
#include <iostream>
#include "../../multiplier.h"

class MultiplierTest : public CxxTest::TestSuite
{
public:

    void testSimple() {
        Multiplier m;
        std::cout << "MultiplierTest " << m.mult("megalomania ", 3) << "\n";
    }
};
