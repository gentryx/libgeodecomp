#ifndef LIBGEODECOMP_IO_IMAGEPAINTER_H
#define LIBGEODECOMP_IO_IMAGEPAINTER_H

#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/storage/image.h>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

/**
 * Adapter class which exhibits an interface compatible with a (very
 * limited) Qt pen.
 */
class ImagePainter
{
public:
    explicit ImagePainter(Image *image) :
        image(image)
    {}

    void moveTo(const Coord<2>& coord)
    {
        position = coord;
    }

    void fillRect(int originX, int originY, int dimX, int dimY, Color color)
    {
        image->fillBox(position + Coord<2>(originX, originY), dimX, dimY, color);
    }

private:
    Image *image;
    Coord<2> position;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
