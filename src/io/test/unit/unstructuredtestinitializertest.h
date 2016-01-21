#include <libgeodecomp/io/unstructuredtestinitializer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredTestInitializerTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        UnstructuredTestInitializer<> initializer(
            100,
            50,
            10);
        std::cout << "boooooooooooooooooomber ==================================================================\n";
        // fixme: needs test
    }
};

}
