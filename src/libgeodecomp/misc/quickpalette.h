#ifndef LIBGEODECOMP_MISC_QUICKPALETTE_H
#define LIBGEODECOMP_MISC_QUICKPALETTE_H

#include <libgeodecomp/misc/color.h>

namespace LibGeoDecomp {

/**
 * This class is similar to Palette, but trades flexibility for speed.
 * Works ony well for floating point numbers.
 */
template<typename VALUE>
class QuickPalette
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;

    explicit QuickPalette(VALUE min = VALUE(), VALUE max = VALUE()) :
        mark0(min),
        mark1((min * 3 + max * 1) / 4),
        mark2((min * 2 + max * 2) / 4),
        mark3((min * 1 + max * 3) / 4),
        mark4(max),
        mult(255.0 * 4 / (max - min))
    {}

    Color operator[](VALUE val) const
    {
        if (val < mark0) {
            return Color::BLACK;
        }

        if (val > mark4) {
            return Color::WHITE;
        }

        if (val < mark1) {
            return Color(0.0, (val - mark0) * mult, 255.0);
        }

        if (val < mark2) {
            return Color(0.0, 255.0, (mark2 - val) * mult);
        }

        if (val < mark3) {
            return Color((val - mark2) * mult, 255.0, 0.0);
        }

        return Color(255.0, (mark4 - val) * mult, 0.0);
    }

private:
    VALUE mark0;
    VALUE mark1;
    VALUE mark2;
    VALUE mark3;
    VALUE mark4;
    VALUE mult;
};

}



#endif
