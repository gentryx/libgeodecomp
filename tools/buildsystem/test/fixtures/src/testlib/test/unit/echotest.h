#include <cxxtest/TestSuite.h>
#include <iostream>

class EchoTest : public CxxTest::TestSuite 
{
public:
    
    void testSimple() {
        std::cout << "EchoTest::testSimple()\n";
    }
};
