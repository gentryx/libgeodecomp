#ifndef LIBGEODECOMP_IO_TESTCELLPLOTTER_H
#define LIBGEODECOMP_IO_TESTCELLPLOTTER_H

#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

class TestCellPlotter
{
public:
    template<typename PAINTER>
    void operator()(
        const TestCell<2>& cell,
        PAINTER painter,
        const Coord<2>& cellDimensions) const
    {
        Color color((int)cell.testValue, 47, 11);
        painter.fillRect(0, 0, cellDimensions.x(), cellDimensions.y(), color);
    }
};

}

#endif
