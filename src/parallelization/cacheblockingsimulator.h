#ifndef _libgeodecomp_parallelization_serialsimulator_h_
#define _libgeodecomp_parallelization_serialsimulator_h_

#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {
/**
 * CacheBlockingSimulator is an experimental simulator to explore the
 * infrastructure required to implement a pipelined wavefront update
 * algorith and which benefits is may provide.
 */
template<typename CELL_TYPE>
class CacheBlockingSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    friend class SerialSimulatorTest;
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIMENSIONS;

    CacheBlockingSimulator(Initializer<CELL_TYPE> *_initializer) : 
        MonolithicSimulator<CELL_TYPE>(_initializer)
    {
        Coord<DIM> dim = initializer->gridBox().dimensions;
        curGrid = new GridType(dim);
        newGrid = new GridType(dim);
        initializer->grid(curGrid);
        initializer->grid(newGrid);
    }
    
    ~CacheBlockingSimulator()
    {
        delete newGrid;
        delete curGrid;
    }

    virtual void step()
    {
        // fixme
    }

    virtual void run()
    {
        // fixme
    }

private:
    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;
    using MonolithicSimulator<CELL_TYPE>::getStep;

    GridType *curGrid;
    GridType *newGrid;

    void updateWavefront(const Coord<DIM>& origin, const Coord<DIM>& dim, unsigned nanoStep, const Coord<DIM>& orginIncrement, const Coord<DIM>& dimDecrement)
    {
        // wavefronts[y][x][z][t]

    }
};

}

#endif

// -1 XXXXXXXXXXXXXXXX
//  0 X00000000000000X
//  1 X00000000000000X
//  2 X00000000000000X
// -----------------------

// 0.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d    ................ 
// e -1 XXXXXXXXXXXXXXXX
// f    ................
// g -1 XXXXXXXXXXXXXXXX
// h  0 X11111111111111X +
// i    ................
// j    ................
// k    ................

// 1.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d    ................ 
// e -1 XXXXXXXXXXXXXXXX
// f    ................
// g -1 XXXXXXXXXXXXXXXX
// h  0 X11111111111111X
// i  1 X11111111111111X +
// j    ................
// k    ................

// 2.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d    ................ 
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X +
// g -1 XXXXXXXXXXXXXXXX
// h  0 X11111111111111X
// i  1 X11111111111111X
// j    ................
// k    ................

// 3.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d    ................ 
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X 
// g -1 XXXXXXXXXXXXXXXX
// h  0 X11111111111111X
// i  1 X11111111111111X
// j  2 X11111111111111X +
// k    ................

// 4.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d    ................ 
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X
// g  1 X22222222222222X +
// h  0 X11111111111111X
// i  1 X11111111111111X
// j  2 X11111111111111X 
// k    ................

// 5.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X +
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X
// g  1 X22222222222222X 
// h  0 X11111111111111X
// i  1 X11111111111111X
// j  2 X11111111111111X
// k    ................

// 6.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  0 X11111111111111X
// i  1 X11111111111111X
// j  2 X11111111111111X
// k  3 X11111111111111X +

// 7.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e -1 XXXXXXXXXXXXXXXX
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  2 X22222222222222X +
// i  1 X11111111111111X
// j  2 X11111111111111X
// k  3 X11111111111111X 

// 8.
// a -1 XXXXXXXXXXXXXXXX
// b    ................ 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e  1 X33333333333333X +
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  1 X11111111111111X
// j  2 X11111111111111X
// k  3 X11111111111111X 

// 9.
// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X +
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  1 X11111111111111X
// j  2 X11111111111111X
// k  3 X11111111111111X 

// 10. pipeline go
// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  1 X11111111111111X
// j  2 X11111111111111X
// k  3 X11111111111111X 
// l  4 X11111111111111X +

// 11.
// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  0 X22222222222222X
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  3 X22222222222222X +
// j  2 X11111111111111X
// k  3 X11111111111111X 
// l  4 X11111111111111X 

// 12.
// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X 
// c -1 XXXXXXXXXXXXXXXX
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  2 X33333333333333X +
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  3 X22222222222222X 
// j  2 X11111111111111X
// k  3 X11111111111111X 
// l  4 X11111111111111X 

// 13.
// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X 
// c  1 X44444444444444X +
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  2 X33333333333333X 
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  3 X22222222222222X 
// j  2 X11111111111111X
// k  3 X11111111111111X 
// l  4 X11111111111111X 

// 14.
// -> to external grid
// -------------------------

// a -1 XXXXXXXXXXXXXXXX
// b  0 X44444444444444X 
// c  1 X44444444444444X +
// d  0 X33333333333333X 
// e  1 X33333333333333X 
// f  2 X33333333333333X 
// g  1 X22222222222222X
// h  2 X22222222222222X 
// i  3 X22222222222222X 
// j  2 X11111111111111X
// k  3 X11111111111111X 
// l  4 X11111111111111X 

