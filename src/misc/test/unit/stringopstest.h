#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/stringops.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StringOpsTest : public CxxTest::TestSuite
{
public:

    void testItoa()
    {
        TS_ASSERT_EQUALS("0",   StringOps::itoa(0));
        TS_ASSERT_EQUALS("-1",  StringOps::itoa(-1));
        TS_ASSERT_EQUALS("123", StringOps::itoa(123));
    }

    void testAtoi()
    {
        TS_ASSERT_EQUALS(0,   StringOps::atoi("0"  ));
        TS_ASSERT_EQUALS(-1,  StringOps::atoi("-1" ));
        TS_ASSERT_EQUALS(123, StringOps::atoi("123"));
    }

    void testTokenize()
    {
        std::string message = "abc_123_andi ist so toll";

        StringOps::StringVec expected1;
        StringOps::StringVec expected2;

        expected1 << "abc"
                  << "123"
                  << "andi ist so toll";

        expected2 << "abc"
                  << "123"
                  << "andi"
                  << "ist"
                  << "so"
                  << "toll";

        TS_ASSERT_EQUALS(expected1, StringOps::tokenize(message, "_"));
        TS_ASSERT_EQUALS(expected2, StringOps::tokenize(message, "_ "));
    }

    void testJoin()
    {
        StringOps::StringVec tokens;
        tokens << "a"
               << "bb"
               << "ccc";
        TS_ASSERT_EQUALS("a--bb--ccc", StringOps::join(tokens, "+-");
        TS_ASSERT_EQUALS("a bb ccc", StringOps::join(tokens, " ");
        TS_ASSERT_EQUALS("abbccc", StringOps::join(tokens, "");
    }
};

}
