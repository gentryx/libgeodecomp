#include <libgeodecomp/config.h>
#include <libgeodecomp/communication/mpilayer.h>

#include <iostream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PTScotchPartitionTest : public CxxTest::TestSuite
{
public:
    void testDummy()
    {
#ifdef LIBGEODECOMP_WITH_SCOTCH

        MPILayer layer;
        std::cout << "layer.rank() = " << layer.rank() << "\n";
        std::cout << "layer.size() = " << layer.size() << "\n";

#endif
    }

};

}
