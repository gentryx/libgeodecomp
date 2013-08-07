#ifndef LIBGEODECOMP_IO_PLOTTER_H
#define LIBGEODECOMP_IO_PLOTTER_H

#include <algorithm>
#ifdef __CODEGEARC__
#include <math.h>
#else
#include <cmath>
#endif
#include <vector>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

template<typename CELL, class CELL_PLOTTER>
class Plotter
{
public:

    /** creates new Plotter object.
     * \param cellPlotter ist used to plott a single cell
     * \param width, height see setCellDimensions()
     */
    Plotter(CELL_PLOTTER *cellPlotter, const unsigned& width = 100, const unsigned& height = 100) :
        cellPlotter(cellPlotter)
    {
        setCellDimensions(width, height);
    }

    /**
     * sets the pixel dimensions of a cell when plotted
     */
    void setCellDimensions(const unsigned& width, const unsigned& height)
    {
        cellDim = Coord<2>(width, height);
    }

    /**
     * \return the dimensions of a cell when plotted (width, height).
     */
    Coord<2> getCellDimensions() const
    {
        return cellDim;
    }

    Image plotGrid(const typename Writer<CELL>::GridType& grid) const
    {
        unsigned width = cellDim.x() * grid.dimensions().x();
        unsigned height = cellDim.y() * grid.dimensions().y();
        return plotGridInViewport(grid, Coord<2>(0, 0), width, height);
    }

    /**
     * Plot the Grid in the given viewport. upperLeft, width and
     * height are pixel coordinates.
     */
    Image plotGridInViewport(
        const typename Writer<CELL>::GridType& grid,
        const Coord<2>& upperLeft,
        const unsigned& width,
        const unsigned& height) const
    {
        Image ret(width, height, Color::BLACK);

        int sx = upperLeft.x() / cellDim.x();
        int sy = upperLeft.y() / cellDim.y();
        int ex = (int)ceil(((double)upperLeft.x() + width)  / cellDim.x());
        int ey = (int)ceil(((double)upperLeft.y() + height) / cellDim.y());
        ex = std::max(ex, 0);
        ey = std::max(ey, 0);
        ex = std::min(ex, grid.dimensions().x());
        ey = std::min(ey, grid.dimensions().y());

        for (int y = sy; y < ey; y++) {
            for (int x = sx; x < ex; x++) {
                Coord<2> relativeUpperLeft =
                    Coord<2>(x * cellDim.x(), y * cellDim.y()) - upperLeft;
                cellPlotter->plotCell(
                    grid.get(Coord<2>(x, y)),
                    &ret,
                    relativeUpperLeft,
                    cellDim.x(),
                    cellDim.y());
            }
        }

        return ret;
    }


private:
    CELL_PLOTTER *cellPlotter;
    Coord<2> cellDim;
};

};

#endif
