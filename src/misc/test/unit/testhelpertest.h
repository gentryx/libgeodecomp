#include <libgeodecomp/misc/testhelper.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TestHelperTest : public CxxTest::TestSuite 
{
    std::string testFile;

public:
    void setUp()
    {
#ifdef WIN32
        testFile = std::string(getenv("WINDIR")) + "\\win.ini";
#else
        testFile = "/etc/hosts";
#endif
    }

    void testAssertFile()
    {
        TS_ASSERT_FILE(testFile);
        TS_ASSERT_NO_FILE(std::string("solarbenite"));
    }

    void testAssertFileContentsEqual()
    {
        TS_ASSERT_FILE_CONTENTS_EQUAL(testFile, testFile);
    }

};

};
