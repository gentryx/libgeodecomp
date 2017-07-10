#ifndef LIBGEODECOMP_MISC_COLOR_H
#define LIBGEODECOMP_MISC_COLOR_H

#include <sstream>

namespace LibGeoDecomp {

/**
 * Our own rudimentary pixel implementation, mostly used to plot 2D
 * simulation data.
 */
class Color
{
 public:
    friend class BoostSerialization;
    friend class HPXSerialization;

    static const Color BLACK;
    static const Color WHITE ;

    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;

    static const Color CYAN;
    static const Color MAGENTA;
    static const Color YELLOW;

    unsigned rgb;

    inline Color() :
        rgb(255u << 24)
    {}

    inline Color(
        const unsigned char r,
        const unsigned char g,
        const unsigned char b) :
        rgb((255 << 24) + (r << 16) + (g << 8) + b)
    {}

    inline Color(
        const char r,
        const char g,
        const char b) :
        rgb((255 << 24) +
            (static_cast<unsigned char>(r) << 16) +
            (static_cast<unsigned char>(g) << 8) +
            static_cast<unsigned char>(b))
    {}

    inline Color(
        const int r,
        const int g,
        const int b) :
        rgb((255 << 24) + (r << 16) + (g << 8) + b)
    {}

    inline Color(
        const double r,
        const double g,
        const double b) :
        rgb((255 << 24) + (int(r) << 16) + (int(g) << 8) + int(b))
    {}

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

    bool operator!=(const Color& com) const
    {
        return !(*this == com);
    }

    std::string toString() const
    {
        std::ostringstream tmp;
        tmp << "Color(" << (int)red() << ", " << (int)green() << ", " << (int)blue() << ")";
        return tmp.str();
    }
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const Color& color)
{
    __os << color.toString();
    return __os;
}

}

#endif
