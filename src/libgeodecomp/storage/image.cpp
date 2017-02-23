#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/storage/image.h>
#include <sstream>

namespace LibGeoDecomp {

namespace ImageHelpers {

void copy(
    const Coord<2>& upperLeftSource,
    const Image& source,
    const Coord<2>& upperLeftTarget,
    Image* target,
    const int width,
    const int height)
{
    const Coord<2>& uls = upperLeftSource;
    const Coord<2>& ult = upperLeftTarget;

    CoordBox<2> sourceRect(
        Coord<2>(0, 0),
        Coord<2>(source.getDimensions().x(), source.getDimensions().y()));
    CoordBox<2> targetRect(
        Coord<2>(0, 0),
        Coord<2>(target->getDimensions().x(), target->getDimensions().y()));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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

            target->set(cTarget, source.get(cSource));
        }
    }
}

}

Image Image::slice(
    const Coord<2>& upperLeft,
    const int width,
    const int height)
{
    Image ret(width, height);
    ImageHelpers::copy(
        upperLeft,
        *this,
        Coord<2>(0,0),
        &ret,
        width,
        height);
    return ret;
}


Image Image::slice(
    const int x,
    const int y,
    const int width,
    const int height)
{
    return slice(Coord<2>(x, y), width, height);
}


void Image::paste(
    const Coord<2>& upperLeft,
    const Image& img)
{
    ImageHelpers::copy(
        Coord<2>(0,0),
        img,
        upperLeft,
        this,
        img.getDimensions().x(),
        img.getDimensions().y());
}


void Image::paste(const int x, const int y, const Image& img)
{
    paste(Coord<2>(x, y), img);
}


}
