#ifndef _libgeodecomp_examples_latticegas_bigcell_h_
#define _libgeodecomp_examples_latticegas_bigcell_h_

#include <libgeodecomp/examples/latticegas/cell.h>

class BigCell
{
public:
    Cell& operator[](const int& y)
    {
        return cells[y];
    }
    
    const Cell& operator[](const int& y) const
    {
        return cells[y];
    }

    void update(const BigCell *up, const BigCell *same, const BigCell *down)
    {
        // fixme: seed
        int seed = 0;
        cells[0].update(seed, 
                        same[ 0][0].getState(),
                        up[   0][1][Cell::LR],
                        up[   1][1][Cell::LL],
                        same[-1][0][Cell::R],
                        same[ 0][0][Cell::C],
                        same[ 1][0][Cell::L],
                        same[ 0][1][Cell::UR],
                        same[ 1][1][Cell::UL]);
        cells[1].update(seed, 
                        same[ 0][1].getState(),
                        same[-1][0][Cell::LR],
                        same[ 0][0][Cell::LL],
                        same[-1][1][Cell::R],
                        same[ 0][1][Cell::C],
                        same[ 1][1][Cell::L],
                        down[-1][0][Cell::UR],
                        down[ 0][0][Cell::UL]);
    } 

    Cell cells[2];
};

#endif
