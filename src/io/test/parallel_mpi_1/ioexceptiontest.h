#include <cerrno>
#include <cxxtest/TestSuite.h>
#include "../../ioexception.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class IOExceptionTest : public CxxTest::TestSuite
{

public:

    void testIOException()
    {
        IOException e("Cannot open file", "/no/such/file", ENOENT);
        TS_ASSERT_EQUALS(std::string("Cannot open file"),
                         std::string(e.what()));
        TS_ASSERT_EQUALS("/no/such/file", e.file());
        TS_ASSERT_EQUALS(ENOENT, e.error());
        TS_ASSERT(e.fatal());
        TS_ASSERT_EQUALS("Cannot open file `/no/such/file': No such file or directory",
                         e.toString());
    }


    void testExceptionWithoutErrorcode()
    {
        IOException e("internal error on", "/some/file");
        TS_ASSERT_EQUALS("internal error on `/some/file'", e.toString());
    }

};

};
