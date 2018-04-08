#ifndef LIBGEODECOMP_IO_PLOTTER_H
#define LIBGEODECOMP_IO_PLOTTER_H

#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/math.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/image.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <algorithm>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * This class renders a 2D grid of cells by stitching together the 2D
 * tiles generated per cell by the CELL_PLOTTER. Useful for generating
 * output images in a Writer, e.g. PPMWriter.
 */
template<typename CELL, class CELL_PLOTTER = SimpleCellPlotter<CELL> >
class Plotter
{
public:
    friend class PPMWriterTest;

    Plotter(
        const Coord<2>& cellDim,
        const CELL_PLOTTER& cellPlotter) :
	cellDim(cellDim),
        cellPlotter(cellPlotter)
    {}

    virtual ~Plotter()
    {}

    template<typename PAINTER>
    void plotGrid(const typename Writer<CELL>::GridType& grid, PAINTER& painter) const
    {
        CoordBox<2> viewport(
            Coord<2>(0, 0),
            Coord<2>(cellDim.x() * grid.dimensions().x(),
                     cellDim.y() * grid.dimensions().y()));
        plotGridInViewport(grid, painter, viewport);
    }

    /**
     * plots a part of the grid. Dimensions are given in pixel
     * coordinates. This is useful to e.g. render a panned excerpt.
     */
    template<typename PAINTER>
    void plotGridInViewport(
        const typename Writer<CELL>::GridType& grid,
        PAINTER& painter,
        const CoordBox<2>& viewport) const
    {
        int sx = viewport.origin.x() / cellDim.x();
        int sy = viewport.origin.y() / cellDim.y();
        int ex = int(ceil((double(viewport.origin.x()) + viewport.dimensions.x()) / cellDim.x()));
        int ey = int(ceil((double(viewport.origin.y()) + viewport.dimensions.y()) / cellDim.y()));
        ex = (std::min)(ex, grid.dimensions().x());
        ey = (std::min)(ey, grid.dimensions().y());

        for (int y = sy; y < ey; ++y) {
            for (int x = sx; x < ex; ++x) {
                Coord<2> relativeUpperLeft =
                    Coord<2>(x * cellDim.x(),
                             y * cellDim.y()) - viewport.origin;
                painter.moveTo(relativeUpperLeft);
                cellPlotter(
                    grid.get(Coord<2>(x, y)),
                    painter,
                    cellDim);
            }
        }
    }

    const Coord<2>& getCellDim()
    {
        return cellDim;
    }

    Coord<2> calcImageDim(const Coord<2>& gridDim)
    {
        return cellDim.scale(gridDim);
    }

private:
    Coord<2> cellDim;
    CELL_PLOTTER cellPlotter;
};

}

#endif
