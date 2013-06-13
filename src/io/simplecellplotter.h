#ifndef LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H
#define LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H

#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/initializer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename CELL_TO_COLOR>
class SimpleCellPlotter
{
public:
    void plotCell(
        const CELL_TYPE& cell, 
        Image *image,
        const Coord<2>& origin, 
        const unsigned& width, 
        const unsigned& height)
    {
        image->paste(origin, Image(width, height, CELL_TO_COLOR()(cell)));
    }    
};

};

#endif
