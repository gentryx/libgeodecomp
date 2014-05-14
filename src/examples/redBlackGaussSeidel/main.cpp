/**
 * 2D Red-Black-Gauss-Sidel example
 * solving a heat transfer problem with Dirichlet-boundary cells
 */
#include <iostream>
#include <cmath>

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

enum CellType {RED, BLACK, BOUNDARY};

class Cell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<2, 1> >,
        public APITraits::HasNanoSteps<2>
    {};

    inline Cell() :
        temp(0), type(BOUNDARY)
    {}

    inline Cell(CellType cellType, double v = 0) :
        temp(v), type(cellType)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
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
 * set bounderi to sin(PI*x)*sinh(PI*y)
 * centerpints to 0
 */
inline double initCellVelu(Coord<2> c, Coord<2> gridDimensions){

        double xPos = ((double)c.x()) / gridDimensions.x();
        double yPos = ((double)c.y()) / gridDimensions.y();

        return sin(M_PI*xPos)*sinh(M_PI*yPos);
}


class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    CellInitializer(const int nx=512, const int ny=512,
                    const unsigned steps=30000)
    : SimpleInitializer<Cell>(Coord<2>(nx, ny), steps)
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
                                   initCellVelu(c0, gridDimensions())) );
            }
            if(bounding.inBounds(c1)){
                ret->set( c1, Cell(BOUNDARY,
                                   initCellVelu(c1, gridDimensions())) );
            }
        }
        for (int y = 0; y < gridDimensions().y(); ++y) {
            Coord<2> c0 (1                     , y);
            Coord<2> c1 (gridDimensions().y()-2, y);

            if(bounding.inBounds(c0)){
                ret->set( c0, Cell(BOUNDARY,
                                   initCellVelu(c0, gridDimensions())) );
            }
            if(bounding.inBounds(c1)){
                ret->set( c1, Cell(BOUNDARY,
                                   initCellVelu(c1, gridDimensions())) );
            }
        }

        // Red Cells
        for (int y = 1; y < gridDimensions().y()-1; ++y){
            for (int x = 1+y%2; x < gridDimensions().x()-1; x+=2){
                Coord<2> c (x,y);

                ret->set( c, Cell(RED) );
            }
        }

        // Black Cells
        for (int y = 1; y < gridDimensions().y()-1; ++y){
            for (int x = 2-y%2; x < gridDimensions().x()-1; x+=2){
                Coord<2> c (x,y);

                ret->set( c, Cell(BLACK) );
            }
        }
    }
};

class CellToColor {
public:
    Color operator()(const Cell& cell)
    {
        if (cell.temp < 0) {
            return Color(0, 0, 0);
        }
        if (cell.temp <= 2.5) {
            return Color(0, (cell.temp - 0.0) * 102, 255);
        }
        if (cell.temp <= 5.0) {
            return Color(0, 255, 255 - (cell.temp - 2.5) * 102);
        }
        if (cell.temp <= 7.5) {
            return Color((cell.temp - 5.0) * 102, 255, 0);
        }
        if (cell.temp <= 10.0) {
            return Color(255, 255 - (cell.temp - 7.5) * 102, 0);
        }
        return Color(255, 255, 255);
    }
};

void runSimulation()
{
    SerialSimulator<Cell> sim(new CellInitializer());

    int outputFrequency = 100;
    sim.addWriter(
        new PPMWriter<Cell, SimpleCellPlotter<Cell, CellToColor> >(
            "gausSidel", outputFrequency, 1, 1)
        );
    sim.addWriter(new TracingWriter<Cell>(outputFrequency, 100));

    sim.run();
}

int main(int argc, char **argv)
{

    runSimulation();

    return 0;
}
