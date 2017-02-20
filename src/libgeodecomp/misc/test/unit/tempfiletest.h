#include <fstream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/tempfile.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TempFileTest : public CxxTest::TestSuite
{
public:

    void assertNoFile(std::string filename)
    {
        std::ifstream input(filename.c_str());
        TSM_ASSERT("File " + filename + " shouldn't exist", !input);
    }


    void testUniqueName()
    {
        std::string prefix = "foo";
        TS_ASSERT_DIFFERS(TempFile::serial(prefix), TempFile::serial(prefix));
    }


    void testGeneratesSuitableNameForOpeningFileAndWriting()
    {
        std::string filename = TempFile::serial("bar");
        assertNoFile(filename);
        std::ofstream test(filename.c_str());
        test << "lorem ipsum etc.";
        test.close();
        remove(filename.c_str());
    }
};

};
