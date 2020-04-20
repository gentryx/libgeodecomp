#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/image.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ImageTest : public CxxTest::TestSuite
{
public:
    void testWidthAndHeight() {
        int width  = 123;
        int height = 234;
        Image img(width, height);
        TS_ASSERT_EQUALS(img.getDimensions().x(), width);
        TS_ASSERT_EQUALS(img.getDimensions().y(), height);
    }

    void testSlice()
    {
        int bigWidth = 123;
        int bigHeight = 234;
        Image big(bigWidth, bigHeight);
        for (int y = 0; y < bigHeight; y++) {
            for (int x = 0; x < bigWidth; x++) {
                big[Coord<2>(x, y)] = Color(x, y, 47);
            }
        }

        int smallWidth = 30;
        int smallHeight = 40;
        Image small(30, 40);
        for (int y = 0; y < smallHeight; y++) {
            for (int x = 0; x < smallWidth; x++) {
                small[Coord<2>(x, y)] = Color(x+10, y+20, 47);
            }
        }

        TS_ASSERT_EQUALS(small, big.slice(Coord<2>(10, 20), 30, 40));
    }

    void testPaste()
    {
        Image a(10, 20, Color::RED);
        Image b(30, 20, Color::YELLOW);
        Image c(40, 20, Color::RED);

        c.paste(10, 0, b);
        TS_ASSERT_EQUALS(a, c.slice( 0, 0, 10, 20));
        TS_ASSERT_EQUALS(b, c.slice(10, 0, 30, 20));
    }

    void testPasteOffScreenLeftUpper()
    {
        Image a(10, 20, Color::RED);
        Image b(40, 30, Color::YELLOW);

        b.paste(-5, -3, a);
        TS_ASSERT_EQUALS(Image(5, 17, Color::RED), b.slice( 0, 0, 5, 17));
    }

    void testPasteOffScreenRightLower()
    {
        Image a(10, 20, Color::RED);
        Image b(40, 30, Color::YELLOW);

        b.paste(37, 15, a);
        TS_ASSERT_EQUALS(Image(3, 15, Color::RED), b.slice(37, 15, 3, 15));
    }

    void testIllegalSliceUpperLeft()
    {
        Image a(10, 20, Color::RED);
        TS_ASSERT_THROWS(a.slice(Coord<2>(-5, -5), 10, 10), std::invalid_argument&);
    }

    void testIllegalSliceLowerRight()
    {
        Image a(10, 20, Color::RED);
        TS_ASSERT_THROWS(a.slice(Coord<2>(5, 15), 10, 10), std::invalid_argument&);
    }

    void testFillBox()
    {
        Image a(10, 10, Color::RED);
        a.fillBox(Coord<2>(-3, 4), 7, 11, Color::YELLOW);

        Image b(10, 10, Color::RED);
        b.paste(0, 4, Image(4, 6, Color::YELLOW));

        TS_ASSERT_EQUALS(a, b);
    }
};

}
