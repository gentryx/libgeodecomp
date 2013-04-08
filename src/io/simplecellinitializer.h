#ifndef LIBGEODECOMP_IO_SIMPLECELLINITIALIZER_H
#define LIBGEODECOMP_IO_SIMPLECELLINITIALIZER_H

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/simplecell.h>

namespace LibGeoDecomp {

class SimpleCellInitializer : public SimpleInitializer<SimpleCell>
{
public:
    SimpleCellInitializer(
        Coord<2> dimensions = Coord<2>(100, 100), 
        const unsigned steps = 300) :
        SimpleInitializer<SimpleCell>(dimensions, steps) 
    {}

    
    virtual Grid<SimpleCell> grid(const CoordBox<2>& box)
    {
        Grid<SimpleCell> grid(box.dimensions);
        for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            double index = 1 + i->x() + i->y() * gridDimensions().x();
            Coord<2> local = *i - box.origin;
            grid[local] = SimpleCell(index);
        }
        grid.getEdgeCell() = SimpleCell(-1);
        return grid;
    }
};

}

#endif
