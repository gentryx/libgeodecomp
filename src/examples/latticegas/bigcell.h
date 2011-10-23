#ifndef _libgeodecomp_examples_latticegas_bigcell_h_
#define _libgeodecomp_examples_latticegas_bigcell_h_

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
            same[ 0][0].getState(),
            up[   0][1][Cell::LR],
            up[   1][1][Cell::LL],
            same[-1][0][Cell::R],
            same[ 0][0][Cell::C],
            same[ 1][0][Cell::L],
            same[ 0][1][Cell::UR],
            same[ 1][1][Cell::UL]);
        cells[1].update(
            simParams,
            t, 
            same[ 0][1].getState(),
            same[-1][0][Cell::LR],
            same[ 0][0][Cell::LL],
            same[-1][1][Cell::R],
            same[ 0][1][Cell::C],
            same[ 1][1][Cell::L],
            down[-1][0][Cell::UR],
            down[ 0][0][Cell::UL]);
    } 

    __device__ __host__ unsigned toColor(SimParams *simParams)
    {
        unsigned r = 0;
        unsigned g = 0;
        unsigned b = 0;

        for (int y = 0; y < 2; ++y) {
            if (cells[y].state != Cell::liquid) {
                r += 255;
                g += 255;
                b += 255;
            } else {
                for (int i = 0; i < 7; ++i) {
                    int col = cells[y].particles[i];
                    r += simParams->palette[col][0];
                    g += simParams->palette[col][1];
                    b += simParams->palette[col][2];
                }
            }
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
