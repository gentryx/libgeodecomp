/**
 * 2D Red-Black-Gauss-Sidel example
 * solving a heat transfer problem with Dirichlet-boundary cells
 */
#include <libgeodecomp.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/math.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iostream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

using namespace LibGeoDecomp;

enum CellType {RED, BLACK, BOUNDARY};

class Cell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<2, 1> >,
        public APITraits::HasNanoSteps<2>,
        public APITraits::HasOpaqueMPIDataType<Cell>
    {};

    explicit
    inline Cell(CellType cellType = BOUNDARY, double v = 0) :
        temp(v),
        type(cellType)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, unsigned nanoStep)
    {
        *this = neighborhood[Coord<2>( 0, 0 )];

        //update RED in fist nanoStep
        if (type == RED && nanoStep == 0){
            temp = (neighborhood[Coord<2>( 0, -1 )].temp +
                    neighborhood[Coord<2>( 0, +1 )].temp +
                    neighborhood[Coord<2>(-1,  0 )].temp +
                    neighborhood[Coord<2>(+1,  0 )].temp
                    ) * (1./4.);
        }
        //update Black in secound nanoStep
        if (type == BLACK && nanoStep == 1){
            temp = (neighborhood[Coord<2>( 0, -1 )].temp +
                    neighborhood[Coord<2>( 0, +1 )].temp +
                    neighborhood[Coord<2>(-1,  0 )].temp +
                    neighborhood[Coord<2>(+1,  0 )].temp
                    ) * (1./4.);
        }
    }

    double temp;
    CellType type;

};

/**
 * range x=[0;1] y[0;1]
 *
 * set boundary to sin(PI*x)*sinh(PI*y),
 * centerpoints to 0
 */
inline double initCellValue(Coord<2> c, Coord<2> gridDimensions){

        double xPos = ((double)c.x()) / gridDimensions.x();
        double yPos = ((double)c.y()) / gridDimensions.y();

        return sin(LIBGEODECOMP_PI * xPos) * sinh(LIBGEODECOMP_PI * yPos);
}


class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    explicit
    CellInitializer(
        const int nx = 512,
        const int ny = 512,
        const unsigned steps = 30000) :
        SimpleInitializer<Cell>(Coord<2>(nx, ny), steps)
    {}

    virtual void grid(GridBase<Cell, 2> *ret)
    {
        CoordBox<2>bounding = ret->boundingBox();

        // Boundary Cells
        for (int x = 0; x < gridDimensions().x(); ++x) {
            Coord<2> c0 (x, 0                     );
            Coord<2> c1 (x, gridDimensions().x()-1);

            if(bounding.inBounds(c0)){
                ret->set( c0, Cell(BOUNDARY,
                                   initCellValue(c0, gridDimensions())) );
            }
            if(bounding.inBounds(c1)){
                ret->set( c1, Cell(BOUNDARY,
                                   initCellValue(c1, gridDimensions())) );
            }
        }
        for (int y = 0; y < gridDimensions().y(); ++y) {
            Coord<2> c0 (1                     , y);
            Coord<2> c1 (gridDimensions().y()-2, y);

            if(bounding.inBounds(c0)){
                ret->set( c0, Cell(BOUNDARY,
                                   initCellValue(c0, gridDimensions())) );
            }
            if(bounding.inBounds(c1)){
                ret->set( c1, Cell(BOUNDARY,
                                   initCellValue(c1, gridDimensions())) );
            }
        }

        // Red Cells
        for (int y = 1; y < gridDimensions().y()-1; ++y){
            for (int x = 1+y%2; x < gridDimensions().x()-1; x+=2){
                Coord<2> c (x,y);

                if(bounding.inBounds(c)){
                    ret->set( c, Cell(RED) );
                }
            }
        }

        // Black Cells
        for (int y = 1; y < gridDimensions().y()-1; ++y){
            for (int x = 2-y%2; x < gridDimensions().x()-1; x+=2){
                Coord<2> c (x,y);

                if(bounding.inBounds(c)){
                    ret->set( c, Cell(BLACK) );
                }
            }
        }
    }
};

void runSimulation()
{
    CellInitializer *init = new CellInitializer(256,256,30000);
    StripingSimulator<Cell> sim(init,
        MPILayer().rank() ? 0 : new NoOpBalancer());

    int outputFrequency = 100;

    PPMWriter<Cell> *ppmWriter = 0;
    if (MPILayer().rank() == 0) {
        ppmWriter = new PPMWriter<Cell>(
            &Cell::temp,
            0.0,
            10.0,
            "gaussSeidel",
            outputFrequency,
            Coord<2>(1, 1));
    }

    CollectingWriter<Cell> *ppmAdapter = new CollectingWriter<Cell>(
        ppmWriter);
    sim.addWriter(ppmAdapter);

    outputFrequency = 1000;
    sim.addWriter(new TracingWriter<Cell>(outputFrequency, init->maxSteps() ));


    sim.run();
}

int main(int argc, char **argv)
{
    MPI_Init (&argc, &argv);

    runSimulation();

    MPI_Finalize();
    return 0;
}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
