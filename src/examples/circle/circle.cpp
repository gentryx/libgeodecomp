#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

class CircleCell
{
    friend class CellToColor;
public:   
    typedef Stencils::VonNeumann<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;
    enum State {LIQUID, SOLIDIFYING, SOLID};
    typedef std::pair<double, double> DPair;

    class API : public APIs::Base
    {};

    static inline unsigned nanoSteps() 
    { 
        return 1; 
    }

    CircleCell() :
        state(LIQUID)
    {}

    CircleCell(const DPair& _relativeCenter, const double& _speed, const double _radius=0) :
        state(SOLIDIFYING),
        relativeCenter(_relativeCenter),
        speed(_speed),
        radius(_radius)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned&)
    {
        *this = neighborhood[Coord<2>(0, 0)];
        switch(state) {
        case LIQUID:
            updateLiquid(neighborhood);
            return;
        case SOLIDIFYING:
            updateSolidifying();
            return;
        case SOLID:
            updateSolid();
            return;
        }
    }
    
private:
    State state;
    DPair relativeCenter;
    double speed;
    double radius;

    bool inCircle(const CircleCell& other, const Coord<2>& pos) const
    {
        double deltaX = other.relativeCenter.first  - pos.x();
        double deltaY = other.relativeCenter.second - pos.y();
        return (deltaX * deltaX + deltaY * deltaY) <= (other.radius * other.radius);
    }

    void updateLiquid(const CoordMap<CircleCell>& neighborhood)
    {
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                const CircleCell& neighbor = neighborhood[Coord<2>(x, y)];
                if (neighbor.state != LIQUID && 
                    (inCircle(neighbor, Coord<2>(0 - x, 0 - y)) ||
                     inCircle(neighbor, Coord<2>(1 - x, 0 - y)) ||
                     inCircle(neighbor, Coord<2>(0 - x, 1 - y)) ||
                     inCircle(neighbor, Coord<2>(1 - x, 1 - y))))
                    turnSolidifying(neighbor, x, y);
            }
        }
    }

    void turnSolidifying(const CircleCell& other, const int& deltaX, const int& deltaY)
    {
        state = SOLIDIFYING;
        relativeCenter = other.relativeCenter;
        relativeCenter.first  += deltaX;
        relativeCenter.second += deltaY;
        speed = other.speed;
        radius = other.radius;
        updateSolidifying();
    }

    void updateSolidifying()
    {
        radius += speed;
        if (inCircle(*this, Coord<2>(0, 0)) &&
            inCircle(*this, Coord<2>(1, 0)) &&
            inCircle(*this, Coord<2>(0, 1)) &&
            inCircle(*this, Coord<2>(1, 1)))
            state = SOLID;
    }

    void updateSolid()
    {
        radius += speed;
    }
};

class CircleCellInitializer : public SimpleInitializer<CircleCell>
{
public:
    CircleCellInitializer(
        Coord<2> _dimensions = Coord<2>(100, 100), 
        const unsigned _steps = 300) :
        SimpleInitializer<CircleCell>(_dimensions, _steps)
    {}

    virtual void grid(GridBase<CircleCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        std::vector<std::pair<Coord<2>, CircleCell> > seeds(3);
        seeds[0] = std::make_pair(
            Coord<2>(1 * gridDimensions().x() / 4, 
                     2 * gridDimensions().y() / 3), 
            CircleCell(std::make_pair(0.5, 0.5), 0.2)); 
        seeds[0] = std::make_pair(
            Coord<2>(3 * gridDimensions().x() / 4, 
                     2 * gridDimensions().y() / 3), 
            CircleCell(std::make_pair(0.5, 0.5), 0.2)); 
        seeds[0] = std::make_pair(
            Coord<2>(1 * gridDimensions().x() / 2, 
                     1 * gridDimensions().y() / 3), 
            CircleCell(std::make_pair(0.5, 0.5), 0.4)); 

        for (std::vector<std::pair<Coord<2>, CircleCell> >::iterator i = 
                 seeds.begin(); i != seeds.end(); ++i) 
            if (rect.inBounds(i->first))
                ret->at(i->first) = i->second;        
    }
};

class CellToColor
{
public:
    Color operator()(const CircleCell& cell)
    {
        if (cell.state == CircleCell::SOLID) 
            return Color::RED;
        if (cell.state == CircleCell::SOLIDIFYING)
            return Color::GREEN;
        return Color::BLACK;
    }
};

void runSimulation()
{
    int outputFrequency = 1;
    SerialSimulator<CircleCell> sim(new CircleCellInitializer());
    new PPMWriter<CircleCell, SimpleCellPlotter<CircleCell, CellToColor> >(
        "./circle", 
        &sim,
        outputFrequency,
        8,
        8);
    new TracingWriter<CircleCell>(&sim, 1);

    sim.run();
}

int main(int, char *[])
{
    runSimulation();
    return 0;
}
