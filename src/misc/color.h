#ifndef LIBGEODECOMP_IO_COLOR_H
#define LIBGEODECOMP_IO_COLOR_H

#include <sstream>

namespace LibGeoDecomp {

class Color
{
 public:
    static const Color BLACK;
    static const Color WHITE ;

    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;

    static const Color CYAN;
    static const Color MAGENTA;
    static const Color YELLOW;

    unsigned rgb;

    inline Color() { *this = Color(0, 0, 0); }

    inline Color(
        const unsigned char& r, 
        const unsigned char& g, 
        const unsigned char& b)
    {
        rgb = 255;
        rgb <<= 8;
        rgb += r;
        rgb <<= 8;
        rgb += g;
        rgb <<= 8;
        rgb += b;
    }

    unsigned char red() const
    {
        return (rgb >> 16) % 256;
    }

    unsigned char green() const
    {
        return (rgb >> 8) % 256;
    }

    unsigned char blue() const
    {
        return (rgb >> 0) % 256;
    }

    bool operator==(const Color& com) const
    {
        return rgb == com.rgb;;
    }

    std::string toString()
    {
        std::ostringstream tmp;
        tmp << "Color(" << (int)red() << ", " << (int)green() << ", " << (int)blue() << ")";
        return tmp.str();
    }
};

};

#endif
