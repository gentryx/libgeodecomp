/**
 * 2D Gaus Sidel example
 */
#include <iostream>
#include <cmath>

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<2, 1> >
    {};

    inline Cell(double v = 0, bool in = true) :
        temp(v), innerCell(in)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        if (neighborhood[Coord<2>( 0, 0 )].innerCell){
            temp = (neighborhood[Coord<2>( 0, -1 )].temp +
                    neighborhood[Coord<2>( 0, +1 )].temp +
                    neighborhood[Coord<2>(-1,  0 )].temp +
                    neighborhood[Coord<2>(+1,  0 )].temp 
                    ) * (1./4.);
            //std::cout << temp << std::endl;
        }
    }

    double temp;
    bool innerCell;

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

		//std::cout << c.x() << " " << c.y() << "; " << gridDimensions.x() << " " << gridDimensions.y() << std::endl;

        return sin(M_PI*xPos)*sinh(M_PI*yPos);
}


class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    CellInitializer(const int nx=512, const int ny=512,
                    const unsigned steps=50000)
    : SimpleInitializer<Cell>(Coord<2>(nx, ny), steps)
    {}

    virtual void grid(GridBase<Cell, 2> *ret)
    {
        CoordBox<2> bounding = ret->boundingBox();

        //std::cout << bounding.toString() << std::endl;
        
        for (int y = 0; y < gridDimensions().y(); ++y){
            for (int x = 0; x < gridDimensions().x(); ++x) {
                Coord<2> c (x,y);

                if (bounding.inBounds(c)){
                    if (x==0 || x==gridDimensions().x()-1 ||
                        y==0 || y==gridDimensions().y()-1 ){

                        ret->set(c,
                                 Cell(initCellVelu(c, gridDimensions()), false)
                                );
                    }
                }
				//else
					//std::cout << "ausen\n" << std::endl;
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
    SerialSimulator<Cell>
        sim(new CellInitializer());

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
