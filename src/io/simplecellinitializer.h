#ifndef _libgeodecomp_io_simplecellinitializer_h_
#define _libgeodecomp_io_simplecellinitializer_h_

#include <libgeodecomp/io/simpleinitializer.h>
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

    
    virtual Grid<SimpleCell> grid(const CoordBox<2>& rect)
    {
        Grid<SimpleCell> grid(rect.dimensions);
        for (CoordBoxSequence<2> s = rect.sequence(); s.hasNext();) {
            Coord<2> coord = s.next();
            double i = 1 + coord.x() + coord.y() * gridDimensions().x();
            Coord<2> local = coord - rect.origin;
            grid[local] = SimpleCell(i);
        }
        grid.getEdgeCell() = SimpleCell(-1);
        return grid;
    }
};

}

#endif
