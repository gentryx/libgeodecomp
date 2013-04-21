#ifndef LIBGEODECOMP_EXAMPLES_LATTICEGAS_BIGCELL_H
#define LIBGEODECOMP_EXAMPLES_LATTICEGAS_BIGCELL_H

#include <libgeodecomp/examples/latticegas/cell.h>

class BigCell
{
public:
    __device__ __host__ Cell& operator[](const int& y)
    {
        return cells[y];
    }
    
    __device__ __host__ const Cell& operator[](const int& y) const
    {
        return cells[y];
    }

    __device__  __host__ void update(
        SimParams *simParams,
        const int& t, 
        const BigCell *up, const BigCell *same, const BigCell *down)
    {
        cells[0].update(
            simParams,
            t, 
            up[   0][1],
            up[   1][1],
            same[-1][0],
            same[ 0][0],
            same[ 1][0],
            same[ 0][1],
            same[ 1][1]);

        cells[1].update(
            simParams,
            t, 
            same[-1][0],
            same[ 0][0],
            same[-1][1],
            same[ 0][1],
            same[ 1][1],
            down[-1][0],
            down[ 0][0]);
    } 

    // fixme: return two unsigneds here in order to render both cells
    __device__ __host__ unsigned toColor(SimParams *simParams)
    {
        unsigned r = 0;
        unsigned g = 0;
        unsigned b = 0;

        for (int y = 0; y < 2; ++y) {
            if (cells[y].state != Cell::liquid) {
                int col = cells[y].state;
                r = simParams->palette[col][0];
                // g = simParams->palette[col][1];
                // b = simParams->palette[col][2];
            }
            // } else {
                for (int i = 0; i < 7; ++i) {
                    int col = 7 + cells[y].particles[i];
                    r += simParams->palette[col][0];
                    g += simParams->palette[col][1];
                    b += simParams->palette[col][2];
                }
            // }
        }

        if (r > 255)
            r = 255;
        if (g > 255)
            g = 255;
        if (b > 255)
            b = 255;

        unsigned a = 0xff;
        return 
            (a << 24) +
            (r << 16) +
            (g <<  8) +
            (b <<  0);
    }

    Cell cells[2];
};

#endif
