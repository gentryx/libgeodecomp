#ifndef LIBGEODECOMP_MISC_PALETTE_H
#define LIBGEODECOMP_MISC_PALETTE_H

#include <libgeodecomp/misc/color.h>

#include <stdexcept>

namespace LibGeoDecomp {

/**
 * This class is a simple way to describe a mapping of a range of
 * values to colors.
 *
 * ATTENTION: this code is slow. We got faster, albeit less flexible
 * implementations of the palette concept.
 */
template<typename VALUE>
class Palette
{
public:
    void addColor(const VALUE& val, const Color& color)
    {
        colors[val] = color;
    }

    Color operator[](const VALUE& val) const
    {
        if (colors.size() == 0) {
            throw std::logic_error("no colors defined");
        }

        typename std::map<VALUE, Color>::const_iterator upperBound = colors.upper_bound(val);
        if (upperBound == colors.end()) {
            --upperBound;
            return upperBound->second;
        }
        if (upperBound == colors.begin()) {
            return upperBound->second;
        }

        typename std::map<VALUE, Color>::const_iterator lowerBound = upperBound;
        --lowerBound;

        VALUE delta = upperBound->first - lowerBound->first;
        VALUE offset1 = val - lowerBound->first;
        VALUE offset2 = delta - offset1;

        Color col(
            (lowerBound->second.red()   * offset2 +
             upperBound->second.red()   * offset1) / delta,
            (lowerBound->second.green() * offset2 +
             upperBound->second.green() * offset1) / delta,
            (lowerBound->second.blue()  * offset2 +
             upperBound->second.blue()  * offset1) / delta);

        return col;
    }

private:
    std::map<VALUE, Color> colors;
};

}

#endif
