#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchlink.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class PatchLinkTest : public CxxTest::TestSuite
{
public:
    void testBasic() 
    {
        std::cout << "------------- LINK -------------\n";
    }
};

}
}
