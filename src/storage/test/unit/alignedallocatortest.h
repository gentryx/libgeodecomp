#include <fstream>
#include <sstream>
#include <cstdio>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/alignedallocator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class AlignedAllocatorTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        {
            int *p = AlignedAllocator<int,   64>().allocate(3);
            TS_ASSERT_EQUALS(0, long(p) %  64);
            AlignedAllocator<int, 64>().deallocate(p, 3);
        }
        {
            char *p = AlignedAllocator<char, 128>().allocate(199);
            TS_ASSERT_EQUALS(0, long(p) % 128);
            AlignedAllocator<char, 128>().deallocate(p, 199);
        }
        {
            long *p = AlignedAllocator<long, 512>().allocate(256);
            TS_ASSERT_EQUALS(0, long(p) % 512);
            AlignedAllocator<long, 512>().deallocate(p, 256);
        }
    }
};

}
