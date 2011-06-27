#include <cxxtest/TestSuite.h>
#include "../../supermap.h"
#include "../../commontypedefs.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class SuperMapTest : public CxxTest::TestSuite 
{
public:
    void testOperatorLessLess()
    {
        SuperMap<int, int> a;
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        std::ostringstream temp;
        temp << a;
        TS_ASSERT_EQUALS("{0 => 1, 1 => 2, 2 => 3}", temp.str());
    }
};

};
