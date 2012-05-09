#ifndef _libgeodecomp_io_plotter_h_
#define _libgeodecomp_io_plotter_h_

#include <algorithm>
#ifdef __CODEGEARC__
#include <math.h>
#else
#include <cmath>
#endif
#include <vector>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/io/image.h>

namespace LibGeoDecomp {

template<typename CELL, class CELL_PLOTTER>
class Plotter 
{
public:

    /** creates new Plotter object. 
     * @param cellPlotter ist used to plott a single cell
     * @param width, height see setCellDimensions()
     */
    Plotter(CELL_PLOTTER *cellPlotter, const unsigned& width = 100, const unsigned& height = 100) :
        _cellPlotter(cellPlotter)
    {
        setCellDimensions(width, height);
    }

    /**
     * sets the pixel dimensions of a cell when plotted 
     */
    void setCellDimensions(const unsigned& width, const unsigned& height)
    {
        _cellWidth = width;
        _cellHeight = height;
    }

    /**
     * @return the dimensions of a cell when plotted (width, height).
     */
    Coord<2> getCellDimensions() const
    {
        return Coord<2>(_cellWidth, _cellHeight);
    }
    
    Image plotGrid(const Grid<CELL, typename CELL::Topology>& grid) const
    {
        unsigned width = _cellWidth * grid.getDimensions().x();
        unsigned height = _cellHeight * grid.getDimensions().y();
        return plotGridInViewport(grid, Coord<2>(0, 0), width, height);
    }

    /**
     * Plot the Grid in the given viewport. upperLeft, width and
     * height are pixel coordinates. 
     */
    Image plotGridInViewport(
        const Grid<CELL, typename CELL::Topology>& grid, 
        const Coord<2>& upperLeft,
        const unsigned& width, 
        const unsigned& height) const
    {
        Image ret(width, height, Color::BLACK);

        int sx = upperLeft.x() / _cellWidth;
        int sy = upperLeft.y() / _cellHeight;
        int ex = (int)ceil(((double)upperLeft.x() + width)  / _cellWidth);
        int ey = (int)ceil(((double)upperLeft.y() + height) / _cellHeight);
        ex = std::max(ex, 0);
        ey = std::max(ey, 0);
        ex = std::min(ex, (int)grid.getDimensions().x());
        ey = std::min(ey, (int)grid.getDimensions().y());

        for (int y = sy; y < ey; y++) {            
            for (int x = sx; x < ex; x++) {
                Coord<2> relativeUpperLeft = 
                    Coord<2>(x * _cellWidth, y * _cellHeight) - upperLeft; 
                _cellPlotter->plotCell(
                    grid[Coord<2>(x, y)], 
                    &ret, 
                    relativeUpperLeft, 
                    _cellWidth, 
                    _cellHeight);
            }            
        }

        return ret;
    }


private:
    CELL_PLOTTER *_cellPlotter;
    unsigned int _cellWidth;
    unsigned int _cellHeight;
};

};

#endif
