#ifndef LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H
#define LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H

#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/initializer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename CELL_TO_COLOR>
class SimpleCellPlotter
{
public:
    template<typename PAINTER>
    void operator()(
        const CELL_TYPE& cell,
	PAINTER& painter,
        const Coord<2>& cellDimensions) const
    {
        painter.fillRect(
            0, 0,
            cellDimensions.x(), cellDimensions.y(),
            CELL_TO_COLOR()(cell));
    }
};

}

#endif
