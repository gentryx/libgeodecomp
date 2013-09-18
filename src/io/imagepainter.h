#ifndef LIBGEODECOMP_IO_IMAGEPAINTER_H
#define LIBGEODECOMP_IO_IMAGEPAINTER_H

#include <libgeodecomp/io/image.h>
#include <libgeodecomp/misc/color.h>

namespace LibGeoDecomp {

class ImagePainter
{
public:
    ImagePainter(Image *image) :
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

}

#endif
