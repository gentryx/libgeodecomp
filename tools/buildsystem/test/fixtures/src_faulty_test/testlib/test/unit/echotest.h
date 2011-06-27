#include <cxxtest/TestSuite.h>
#include <iostream>

class EchoTest : public CxxTest::TestSuite 
{
public:
    
    void testSimple() 
    {
    }

    void foo(unsigned i, int j)
    {
        // should fail because of unsigned to int comparison and -Wall and -Werror
        if (i < j)
            std::cout << "boom\n";
    }
};
