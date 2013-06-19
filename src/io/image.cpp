#include <sstream>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/misc/coordbox.h>

namespace LibGeoDecomp {

void copy(const Coord<2>& upperLeftSource,
          const Image& source,
          const Coord<2>& upperLeftTarget,
          Image* target,
          const unsigned& width,
          const unsigned& height)
{
    const Coord<2>& uls = upperLeftSource;
    const Coord<2>& ult = upperLeftTarget;
    CoordBox<2> sourceRect(
        Coord<2>(0, 0),
        Coord<2>(source.getDimensions().x(), source.getDimensions().y()));
    CoordBox<2> targetRect(
        Coord<2>(0, 0),
        Coord<2>(target->getDimensions().x(), target->getDimensions().y()));

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            Coord<2> cTarget = ult + Coord<2>(x, y);
            Coord<2> cSource = uls + Coord<2>(x, y);

            // skip off-screen target coords
            if (!targetRect.inBounds(cTarget)) {
                continue;
            }
            if (!sourceRect.inBounds(cSource)) {
                throw std::invalid_argument(
                    "Source coordinate " + cSource.toString() +
                    " is not within " + sourceRect.toString());
            }

            (*target)[cTarget.y()][cTarget.x()] =
                source[cSource.y()][cSource.x()];
        }
    }
}

Image Image::slice(const Coord<2>& upperLeft,
                   const unsigned& width,
                   const unsigned& height)
{
    Image ret(width, height);
    copy(upperLeft, *this, Coord<2>(0,0), &ret, width, height);
    return ret;
}


Image Image::slice(const unsigned& x, const unsigned& y, const unsigned& width, const unsigned& height)
{
    return slice(Coord<2>(x, y), width, height);
}


void Image::paste(const Coord<2>& upperLeft, const Image& img)
{
    copy(Coord<2>(0,0), img, upperLeft, this,
         img.getDimensions().x(), img.getDimensions().y());
}


void Image::paste(const int& x, const int& y, const Image& img)
{
    paste(Coord<2>(x, y), img);
}

}
