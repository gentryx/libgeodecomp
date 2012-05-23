#ifndef _libgeodecomp_io_image_h_
#define _libgeodecomp_io_image_h_

#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/grid.h>

namespace LibGeoDecomp {

class Image : public Grid<Color>
{
 public:
    class IllegalCoordException {};

    inline Image(const unsigned& width, 
                 const unsigned& height, 
                 const Color& col=Color()) :
        Grid<Color>(Coord<2>(width, height), col)
    {}

    Image slice(const Coord<2>& upperLeft, 
                const unsigned& width, 
                const unsigned& height);

    Image slice(const unsigned& x, 
                const unsigned& y, 
                const unsigned& width, 
                const unsigned& height);

    void paste(const Coord<2>& upperLeft, const Image& img);

    void paste(const int& x, const int& y, const Image& img);
    
    inline void setPix(const int& x, const int& y, const Color& col)
    {
        if (x < 0 || y < 0 || 
            x >= (int)getDimensions().x() || y >= (int)getDimensions().y())
            return;
        (*this)[y][x] = col;
    }

    inline void fillBox(const Coord<2>& upperLeft, 
                        const unsigned& boxWidth, 
                        const unsigned& boxHeight, 
                        const Color& col)
    {
        Coord<2> lowerRight = upperLeft + Coord<2>(boxWidth, boxHeight);
        int sx = std::max(upperLeft.x(), 0);
        int sy = std::max(upperLeft.y(), 0);
        int ex = std::min(lowerRight.x(), (int)getDimensions().x());
        int ey = std::min(lowerRight.y(), (int)getDimensions().y());
        
        for (int y = sy; y < ey; y++) 
            for (int x = sx; x < ex; x++) {
                (*this)[y][x] = col;            
        }
    }
};

}

#endif
