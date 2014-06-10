#ifndef LIBGEODECOMP_IO_PLOTTER_H
#define LIBGEODECOMP_IO_PLOTTER_H

#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/image.h>

#include <algorithm>
#ifdef __CODEGEARC__
#include <math.h>
#else
#include <cmath>
#endif

#include <vector>

namespace LibGeoDecomp {

template<typename CELL, class CELL_PLOTTER = SimpleCellPlotter<CELL> >
class Plotter
{
public:
    explicit Plotter(const Coord<2>& cellDim = Coord<2>(32, 32),
                     const CELL_PLOTTER& cellPlotter = CELL_PLOTTER()) :
	cellDim(cellDim),
        cellPlotter(cellPlotter)
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
        int ex = ceil((double(viewport.origin.x()) + viewport.dimensions.x()) / cellDim.x());
        int ey = ceil((double(viewport.origin.y()) + viewport.dimensions.y()) / cellDim.y());
        ex = std::min(ex, grid.dimensions().x());
        ey = std::min(ey, grid.dimensions().y());

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

    Coord<2> calcImageDim(const Coord<2> gridDim)
    {
        return Coord<2>(
            cellDim.x() * gridDim.x(),
            cellDim.y() * gridDim.y());
    }

private:
    Coord<2> cellDim;
    CELL_PLOTTER cellPlotter;
};

}

#endif
