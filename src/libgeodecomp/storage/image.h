#ifndef LIBGEODECOMP_STORAGE_IMAGE_H
#define LIBGEODECOMP_STORAGE_IMAGE_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

/**
 * Simple, stupid utility class to draw images without pulling in
 * external dependencies (e.g. Qt).
 */
class Image : public Grid<Color>
{
 public:
    class IllegalCoordException {};

    inline Image(
        const int width,
        const int height,
        const Color col=Color()) :
        Grid<Color>(Coord<2>(width, height), col)
    {}

    inline explicit Image(
        const Coord<2>& dim,
        const Color col=Color()) :
        Grid<Color>(dim, col)
    {}

    Image slice(
        const Coord<2>& upperLeft,
        const int width,
        const int height);

    Image slice(
        const int x,
        const int y,
        const int width,
        const int height);

    void paste(
        const Coord<2>& upperLeft,
        const Image& img);

    void paste(
        const int x,
        const int y,
        const Image& img);

    inline
    void fillBox(
        const Coord<2>& upperLeft,
        const int boxWidth,
        const int boxHeight,
        const Color& col)
    {
        Coord<2> lowerRight = upperLeft + Coord<2>(boxWidth, boxHeight);
        int sx = (std::max)(upperLeft.x(), 0);
        int sy = (std::max)(upperLeft.y(), 0);
        int ex = (std::min)(lowerRight.x(), (int)getDimensions().x());
        int ey = (std::min)(lowerRight.y(), (int)getDimensions().y());

        for (int y = sy; y < ey; y++) {
            for (int x = sx; x < ex; x++) {
                set(Coord<2>(x, y), col);
            }
        }
    }
};

}

#endif
