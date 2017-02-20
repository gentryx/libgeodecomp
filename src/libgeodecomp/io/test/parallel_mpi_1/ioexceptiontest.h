#include <cerrno>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/ioexception.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class IOExceptionTest : public CxxTest::TestSuite
{
public:

    void testIOException()
    {
        IOException e("Cannot open file /no/such/file");
        TS_ASSERT_EQUALS(
            std::string("Cannot open file /no/such/file"),
            std::string(e.what()));
    }

    void testExceptionWithoutErrorcode()
    {
        IOException e("internal error on /some/file");
        TS_ASSERT_EQUALS(std::string("internal error on /some/file"), e.what());
    }
};

}
