#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_TESTPLOTTER_H
#define LIBGEODECOMP_IO_TESTPLOTTER_H

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/image.h>

namespace LibGeoDecomp {

class TestPlotter 
{
public:
    void plotCell(
        const TestCell<2> & cell, 
        Image *image,
        const Coord<2>& origin, 
        const unsigned& width, 
        const unsigned& height)
    {
        Color color((int)cell.testValue, 47, 11);
        image->paste(origin, Image(width, height, color));
    }
};

};

#endif
#endif
