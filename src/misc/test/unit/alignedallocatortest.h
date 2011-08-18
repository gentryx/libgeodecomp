#include <fstream>
#include <sstream>
#include <cstdio>
#include <cxxtest/TestSuite.h>
#include "../../alignedallocator.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class AlignedAllocatorTest : public CxxTest::TestSuite 
{
public:

    void testBasic()
    {
        TS_ASSERT_EQUALS(
            0, ((long)AlignedAllocator<int,   64>().allocate(3))   %  64);
        TS_ASSERT_EQUALS(
            0, ((long)AlignedAllocator<char, 128>().allocate(199)) % 128);
        TS_ASSERT_EQUALS(
            0, ((long)AlignedAllocator<long, 512>().allocate(256)) % 512);
    }
};

}
