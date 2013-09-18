#include <cmath>
#include <iomanip>
#include <iostream>
#include <silo.h>
#include <stdlib.h>

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

#define NOZZLE 1
#define CHROMO 2
#define SETUP 2

#if SETUP==NOZZLE
const int MAX_X   = 2000;
const int MAX_Y   = 8000;
const int LIQUID_CELLS = 8000;
#endif

#if SETUP==CHROMO
const int MAX_X   = 4000;
const int MAX_Y   = 10000;
const int LIQUID_CELLS = 40000;
#endif

enum State {LIQUID=0, SOLID=1, COAL=2};

const double INFLUXES[] =  {80, 10, 10};
const double ZERO_INFLUXES[] = {0, 0, 0};
const double EFFLUX =  100;

const double PRESSURE_DIFFUSION = 3.0;
const double FLOW_DIFFUSION = 10.0;
const double PRESSURE_SPEED = 1.0;
const double VELOCITY_MOD[] = {1.0, 0.0, 0.93};
const double ABSORBTION_RATES[] = {0.0, 0.1, 0.0};

const int SUBSTANCES = 3;
const int SAMPLES = 1000;
const int ELEMENT_SPACING = 50;
const int CELL_SPACING = 400;

Coord<2> FarAway(-1, -1);

class ID
{
public:
    ID(const Coord<2>& containerCoord=FarAway, const int& index=-1) :
        container(containerCoord),
        num(index)
    {}

    bool operator==(const ID& other) const
    {
        return other.container == container && other.num == num;
    }

    Coord<2> container;
    int num;
};

class Cell
{
public:
    static const int MAX_NEIGHBORS = 20;

    Cell(const Coord<2>& _center=FarAway, const ID& _id=ID(),
         const double _influxes[SUBSTANCES] = ZERO_INFLUXES,
         const double& _efflux = 0,
         const State& _state = LIQUID) :
        state(_state),
        efflux(_efflux),
        pressure(0),
        absorbedVolume(0),
        velocityX(0),
        velocityY(0),
        center(_center),
        id(_id),
        area(0),
        numNeighbors(0),
        shapeSize(0)
    {
        for (int i = 0; i < SUBSTANCES; ++i) {
            quantities[i] = 0;
            ratios[i] = 0;
        }

        std::copy(_influxes, _influxes + SUBSTANCES, influxes);
    }

    template<class CONTAINER>
    void update(CONTAINER *container, const int& nanoStep)
    {
        *this = container->cell(id);

        if (nanoStep == 0) {
            setFluxes(container);
        }
        if (nanoStep == 1) {
            diffuse(container);
        }
    }

    template<class CONTAINER>
    void setFluxes(CONTAINER *container)
    {
        if (state == SOLID)
            return;

        double mass = getMass();
        double totalFlux = 0;

        for (int i = 0; i < numNeighbors; ++i) {
            const Cell& other = container->cell(neighborIDs[i]);
            if (other.state == SOLID) {
                fluxesFlow[i] = 0;
                fluxesPressure[i] = 0;
            } else {
                const Coord<2>& dir = -borderDirections[i];
                double length = 1.0 / sqrt(dir.x() * dir.x() +
                                           dir.y() * dir.y());
                double fluxFlow = (dir.x() * velocityX +
                                   dir.y() * velocityY) *
                    length * pressure * borderLengths[i] * FLOW_DIFFUSION;
                double fluxPressure = 0;
                if (pressure > other.pressure)
                    fluxPressure = (pressure - other.pressure) *
                        borderLengths[i] * PRESSURE_DIFFUSION;
                fluxesFlow[i]     = std::max(0.0, fluxFlow);
                fluxesPressure[i] = fluxPressure;
                totalFlux += fluxesPressure[i] + fluxesFlow[i];
            }
        }

        if (totalFlux > mass) {
            throw std::logic_error("unstable");
        }

        if (mass > 0) {
            double factor = (mass - totalFlux) / mass;
            for (int i = 0; i < SUBSTANCES; ++i)
                quantities[i] *= factor;
        }
    }

    template<class CONTAINER>
    void diffuse(CONTAINER *container)
    {
        if (state == SOLID) {
            pressure = 0;
            return;
        }

        double mass = getMass();
        double fluxes[SUBSTANCES];
        for (int i = 0; i < SUBSTANCES; ++i)
            fluxes[i] = 0;
        double fluxVelocityX = 0;
        double fluxVelocityY = 0;

        for (int i = 0; i < numNeighbors; ++i) {
            const Cell& other = container->cell(neighborIDs[i]);
            if (other.state != SOLID) {
                const Coord<2>& dir = borderDirections[i];
                double length = 1.0 / sqrt(dir.x() * dir.x() +
                                           dir.y() * dir.y());

                for (int j = 0; j < other.numNeighbors; ++j) {
                    if (other.neighborIDs[j] == id) {
                        double otherMass = other.getMass();
                        if (otherMass > 0) {
                            double fluxCoefficient =
                                (other.fluxesPressure[j] + other.fluxesFlow[j]) / otherMass;
                            for (int i = 0; i < SUBSTANCES; ++i)
                                fluxes[i] += fluxCoefficient * other.quantities[i];
                            fluxVelocityX += other.fluxesFlow[j] * other.velocityX;
                            fluxVelocityY += other.fluxesFlow[j] * other.velocityY;
                            double pressureCoefficient =
                                other.fluxesPressure[j] * length * PRESSURE_SPEED;
                            fluxVelocityX += dir.x() * pressureCoefficient;
                            fluxVelocityY += dir.y() * pressureCoefficient;
                        }
                        break;
                    }
                }
            }
        }

        double newMass = mass;
        for (int i = 0; i < SUBSTANCES; ++i) {
            newMass += fluxes[i];
        }

        double scale = VELOCITY_MOD[state] / newMass;
        velocityX = (mass * velocityX + fluxVelocityX) * scale;
        velocityY = (mass * velocityY + fluxVelocityY) * scale;
        for (int i = 0; i < SUBSTANCES; ++i) {
            quantities[i] += fluxes[i] + influxes[i];
        }

        mass = getMass();
        if (mass > 0) {
            double fluxCoefficient = 1.0 - efflux / mass;
            for (int i = 0; i < SUBSTANCES; ++i)
                quantities[i] *= fluxCoefficient;
        }

        for (int i = 0; i < SUBSTANCES; ++i)
            quantities[i] = std::max(0.0, quantities[i]);

        handleAbsorbtion();
        updatePressures();
    }

    void handleAbsorbtion()
    {
        if (state != COAL) {
            return;
        }

        double maxAbsorbtionVolume = area;
        double freeVolume = maxAbsorbtionVolume - absorbedVolume;

        double fillLevel = absorbedVolume / area;
        double freeLevel = 1 - fillLevel;
        double substancePressures[SUBSTANCES];
        for (int i = 0; i < SUBSTANCES; ++i) {
            substancePressures[i] = ratios[i] * pressure;
        }

        double absorbtions[SUBSTANCES];
        for (int i = 0; i < SUBSTANCES; ++i) {
            absorbtions[i] = area * freeLevel * substancePressures[i] * ABSORBTION_RATES[i];
        }
        for (int i = 0; i < SUBSTANCES; ++i) {
            absorbtions[i] = std::min(absorbtions[i], quantities[i]);
            absorbtions[i] = std::min(absorbtions[i], freeVolume);
            freeVolume -= absorbtions[i];
            quantities[i] -= absorbtions[i];
            absorbedVolume += absorbtions[i];
        }
    }

    double getMass() const
    {
        double mass = 0;
        for (int i = 0; i < SUBSTANCES; ++i) {
            mass += quantities[i];
        }

        return mass;
    }

    void updatePressures()
    {
        double mass = getMass();
        pressure = mass / area;
        for (int i = 0; i < SUBSTANCES; ++i) {
            ratios[i] = quantities[i] / mass;
        }
    }

    void pushNeighbor(
        const ID& id,
        const double& length,
        const Coord<2>& dir)
    {
        if (numNeighbors >= MAX_NEIGHBORS) {
            throw std::logic_error("too many neighbors");
        }
        neighborIDs[numNeighbors] = id;
        borderLengths[numNeighbors] = length;
        borderDirections[numNeighbors] = dir;
        ++numNeighbors;
    }

    void setShape(const SuperVector<Coord<2> >& newShape)
    {
        if (newShape.size() > MAX_NEIGHBORS) {
            throw std::logic_error("shape too large");
        }

        shapeSize = newShape.size();
        std::copy(newShape.begin(), newShape.end(), shape);
    }

    State state;
    double quantities[SUBSTANCES];
    double ratios[SUBSTANCES];
    double influxes[SUBSTANCES];
    double efflux;
    double pressure;
    double absorbedVolume;
    double velocityX;
    double velocityY;

    Coord<2> center;
    ID id;
    double area;
    int numNeighbors;
    int shapeSize;

    ID neighborIDs[MAX_NEIGHBORS];
    double borderLengths[MAX_NEIGHBORS];
    double fluxesFlow[MAX_NEIGHBORS];
    double fluxesPressure[MAX_NEIGHBORS];
    Coord<2> borderDirections[MAX_NEIGHBORS];
    Coord<2> shape[MAX_NEIGHBORS];
};

class ContainerCell
{
public:
    const static int MAX_CELLS = 100;

    class API :
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasNanoSteps<2>
    {};

    typedef Grid<ContainerCell, Topology> GridType;
    typedef CoordMap<ContainerCell, GridType> CoordMapType;

    ContainerCell() :
        numCells(0)
    {}

    const Cell& cell(const ID id)
    {
        Coord<2> delta = id.container - coord;
        return (*neighbors)[delta].cells[id.num];
    }

    ContainerCell& operator<<(const Cell& cell)
    {
        if (numCells >= MAX_CELLS) {
            throw std::logic_error("too many cells");
        }

        cells[numCells++] = cell;
        return *this;
    }

    void update(const CoordMapType& neighborhood, const unsigned& nanoStep)
    {
        neighbors = &neighborhood;
        for (int i = 0; i < numCells; ++i) {
            cells[i].update(this, nanoStep);
        }
    }

    Coord<2> coord;
    Cell cells[MAX_CELLS];
    int numCells;
    const CoordMapType *neighbors;
};

class Equation
{
public:
    Equation(const Coord<2>& base, const Coord<2>& dir, ID id = ID()) :
        base(base),
        dir(dir),
        neighborID(id),
        length(-1)
    {}

    bool includes(const Coord<2>& point) const
    {
        return (point - base) * dir > 0;
    }

    Coord<2> base;
    Coord<2> dir;
    ID neighborID;
    double length;
};

class Element
{
public:
    Element(const Coord<2> center = Coord<2>(1, 1), ID id = ID()) :
        center(center),
        id(id)
    {
        limits << Equation(Coord<2>(center.x(), 0),     Coord<2>(0, 1))
               << Equation(Coord<2>(0, center.y()),     Coord<2>(1, 0))
               << Equation(Coord<2>(MAX_X, center.y()), Coord<2>(-1, 0))
               << Equation(Coord<2>(center.x(), MAX_Y), Coord<2>(0, -1));
    }

    Element& operator<<(const Equation& eq)
    {
        limits << eq;
        SuperVector<Coord<2> > cutPoints = generateCutPoints();
        SuperSet<int> deleteSet;

        for (int i = 0; i < limits. size(); ++i) {
            Coord<2> leftDir = turnLeft90(limits[i].dir);
            int dist1 = (cutPoints[2 * i + 0] - limits[i].base) * leftDir;
            int dist2 = (cutPoints[2 * i + 1] - limits[i].base) * leftDir;
            if (dist2 >= dist1) {
                // twisted differences, deleting
                deleteSet.insert(i);
            }

            for (int j = 0; j < limits.size(); ++j)
                if (i != j) {
                    // parallel lines, deleting...
                    if (cutPoint(limits[i], limits[j]) == FarAway) {
                        if (limits[i].dir * limits[j].dir > 0) {
                            int dist1 = (center - limits[i].base) *
                                limits[i].dir;
                            int dist2 = (center - limits[j].base) *
                                limits[i].dir;
                            if (dist2 >= dist1) {
                                deleteSet.insert(j);
                            }
                        }
                    }
                }
        }

        SuperVector<Equation> newLimits;
        for (int i = 0; i < limits.size(); ++i) {
            if (!deleteSet.count(i)) {
                newLimits << limits[i];
            }
        }
        limits = newLimits;

        return *this;
    }

    Element& operator<<(const Element& e)
    {
        Coord<2> base = (center + e.center) / 2;
        Coord<2> dir = center - e.center;
        *this << Equation(base, dir, e.id);
        return *this;
    }

    Element& operator<<(const Cell& c)
    {
        Coord<2> base = (center + c.center) / 2;
        Coord<2> dir = center - c.center;
        *this << Equation(base, dir, c.id);
        return *this;
    }

    SuperVector<Coord<2> > generateCutPoints()
    {
        SuperVector<Coord<2> > buf(2 * limits.size(), FarAway);

        for (int i = 0; i < limits.size(); ++i) {
            for (int j = 0; j < limits.size(); ++j) {
                if (i != j) {
                    Coord<2> cut = cutPoint(limits[i], limits[j]);
                    int offset = 2 * i;
                    Coord<2> delta = cut - limits[i].base;
                    Coord<2> turnedDir = turnLeft90(limits[i].dir);
                    double distance =
                        1.0 * delta.x() * turnedDir.x() +
                        1.0 * delta.y() * turnedDir.y();


                    bool isLeftCandidate = true;
                    if (limits[j].dir * turnLeft90(limits[i].dir) > 0) {
                        isLeftCandidate = false;
                        offset += 1;
                    }

                    delta = buf[offset] - limits[i].base;
                    double referenceDist =
                        1.0 * delta.x() * turnedDir.x() +
                        1.0 * delta.y() * turnedDir.y();
                    bool flag = false;
                    if (buf[offset] == FarAway) {
                        flag = true;
                    }
                    if (isLeftCandidate  && (distance < referenceDist)) {
                        flag = true;
                    }
                    if (!isLeftCandidate && (distance > referenceDist)) {
                        flag = true;
                    }
                    if (cut == FarAway) {
                        flag = false;
                    }
                    if (flag) {
                        buf[offset] = cut;
                    }
                }
            }
        }

        return buf;
    }

    SuperVector<Coord<2> > getShape()
    {
        SuperVector<Coord<2> > cutPoints = generateCutPoints();

        for (int i = 0; i < cutPoints.size(); ++i) {
            if (cutPoints[i] == FarAway) {
                throw std::logic_error("invalid cut point");
            }
        }

        SuperMap<double, Coord<2> > points;
        for (int i = 0; i < cutPoints.size(); ++i) {
            Coord<2> delta = cutPoints[i] - center;
            double length = sqrt(delta.x() * delta.x() + delta.y() * delta.y());
            double dY = delta.y() / length;
            double angle = asin(dY);
            if (delta.x() < 0) {
                angle = M_PI - angle;
            }
            points[angle] = cutPoints[i];
        }

        SuperVector<Coord<2> > res;
        for (SuperMap<double, Coord<2> >::iterator i = points.begin();
             i != points.end(); ++i) {
            res << i->second;
        }

        if (res.size() < 3) {
            throw std::logic_error("cycle too short");
        }

        return res;
    }

    bool includes(const Coord<2>& c)
    {
        for (int i = 0; i < limits.size(); ++i) {
            if (!limits[i].includes(c)) {
                return false;
            }
        }
        return true;
    }

    void updateGeometryData()
    {
        SuperVector<Coord<2> > cutPoints = generateCutPoints();

        for (int i = 0; i < limits. size(); ++i) {
            Coord<2> delta = cutPoints[2 * i + 0] - cutPoints[2 * i + 1];
            limits[i].length = sqrt(delta.x() * delta.x() + delta.y() * delta.y());
        }

        Coord<2> min(MAX_X, MAX_Y);
        Coord<2> max(0, 0);
        for (int i = 0; i < cutPoints.size(); ++i) {
            Coord<2>& c = cutPoints[i];
            max = c.max(max);
            min = c.min(min);
        }
        Coord<2> delta = max - min;

        int hits = 0;
        for (int i = 0; i < SAMPLES; ++i) {
            Coord<2> p = Coord<2>(rand() % delta.x(),
                                  rand() % delta.y()) + min;
            if (includes(p)) {
                ++hits;
            }
        }
        area = 1.0 * hits / SAMPLES * delta.prod();

        diameter = std::max(delta.x(), delta.y());
        if (diameter > CELL_SPACING/2) {
            std::cout << "my diameter: " << diameter << "\n"
                      << "min: " << min << "\n"
                      << "max: " << max << "\n";
            throw std::logic_error("element too large");
        }
    }

    const Coord<2>& getCenter() const
    {
        return center;
    }

    const double& getArea() const
    {
        return area;
    }

    const SuperVector<Equation>& getLimits() const
    {
        return limits;
    }

    const double& getDiameter() const
    {
        return diameter;
    }

private:
    Coord<2> center;
    ID id;
    double area;
    double diameter;
    SuperVector<Equation> limits;

    Coord<2> turnLeft90(const Coord<2>& c)
    {
        return Coord<2>(-c.y(), c.x());
    }

    Coord<2> cutPoint(Equation eq1, Equation eq2)
    {
        if (eq1.dir.y() == 0) {
            if (eq2.dir.y() == 0) {
                // throw std::invalid_argument("both lines are vertical")
                return FarAway;
            }
            std::swap(eq1, eq2);
        }

        Coord<2> dir1 = turnLeft90(eq1.dir);
        double m1 = 1.0 * dir1.y() / dir1.x();
        double d1 = eq1.base.y() - m1 * eq1.base.x();

        if (eq2.dir.y() == 0) {
            return Coord<2>(eq2.base.x(), eq2.base.x() * m1 + d1);
        }

        Coord<2> dir2 = turnLeft90(eq2.dir);
        double m2 = 1.0 * dir2.y() / dir2.x();
        double d2 = eq2.base.y() - m2 * eq2.base.x();

        if (m1 == m2) {
            // throw std::invalid_argument("parallel lines")
            return FarAway;
        }

        double x = (d2 - d1) / (m1 - m2);
        double y = d1 + x * m1;

        if ((x < (-10 * MAX_X)) ||
            (x > ( 10 * MAX_X)) ||
            (y < (-10 * MAX_Y)) ||
            (y > ( 10 * MAX_Y))) {
            return FarAway;
        }

        return Coord<2>(x, y);
    }
};

Coord<2> randCoord()
{
    int x = ELEMENT_SPACING / 2 + rand() % (MAX_X - ELEMENT_SPACING);
    int y = ELEMENT_SPACING / 2 + rand() % (MAX_Y - ELEMENT_SPACING);
    return Coord<2>(x, y);
}

Coord<2> pointToContainerCoord(const Coord<2>& c)
{
    return Coord<2>(c.x() / CELL_SPACING, c.y() / CELL_SPACING);
}

class ChromoInitializer : public LibGeoDecomp::SimpleInitializer<ContainerCell>
{
public:
    ChromoInitializer(
        const Coord<2>& dim,
        const unsigned& steps) :
        SimpleInitializer<ContainerCell>(dim, steps)
    {}

    virtual void grid(GridBase<ContainerCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        ret->atEdge() = ContainerCell();

        Grid<ContainerCell> grid = createBasicGrid();
        fillGeometryData(&grid);

        for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            ret->at(*i) = grid[*i];
            ret->at(*i).coord = *i;
        }
    }

private:

    Grid<ContainerCell> createBasicGrid()
    {
        srand(0);
        Coord<2> cellDim(ceil(1.0 * MAX_X / CELL_SPACING),
                         ceil(1.0 * MAX_Y / CELL_SPACING));
        Grid<ContainerCell> grid(cellDim);

#if SETUP==NOZZLE
        addInletOutlet(&grid);
        addNozzle(&grid);
#endif

#if SETUP==CHROMO
        addInletOutlet(&grid);
        addCoal(&grid, MAX_X * 0.05, MAX_Y * 0.10, MAX_X * 0.10);
        addCoal(&grid, MAX_X * 0.10, MAX_Y * 0.16, MAX_X * 0.18);
        addCoal(&grid, MAX_X * 0.35, MAX_Y * 0.20, MAX_X * 0.12);
        addCoal(&grid, MAX_X * 0.40, MAX_Y * 0.16, MAX_X * 0.21);

        addCoal(&grid, MAX_X * 0.10, MAX_Y * 0.30, MAX_X * 0.15);
        addCoal(&grid, MAX_X * 0.30, MAX_Y * 0.29, MAX_X * 0.08);
        addCoal(&grid, MAX_X * 0.43, MAX_Y * 0.27, MAX_X * 0.07);
        addCoal(&grid, MAX_X * 0.50, MAX_Y * 0.23, MAX_X * 0.11);
        addCoal(&grid, MAX_X * 0.60, MAX_Y * 0.32, MAX_X * 0.15);

        addCoal(&grid, MAX_X * 0.90, MAX_Y * 0.45, MAX_X * 0.18);
        addCoal(&grid, MAX_X * 0.80, MAX_Y * 0.58, MAX_X * 0.18);
        addCoal(&grid, MAX_X * 0.57, MAX_Y * 0.61, MAX_X * 0.18);
        addCoal(&grid, MAX_X * 0.60, MAX_Y * 0.49, MAX_X * 0.15);
        addCoal(&grid, MAX_X * 0.30, MAX_Y * 0.47, MAX_X * 0.20);

        addCoal(&grid, MAX_X * 0.05, MAX_Y * 0.94, MAX_X * 0.15);
        addCoal(&grid, MAX_X * 0.41, MAX_Y * 0.90, MAX_X * 0.21);

        srand(0);
        for (int i = 0; i < 30; ++i) {
            Coord<2> c = randCoord();
            c.y() = MAX_Y * 0.8 + c.y() * 0.1;
            double radius = (0.05 + (rand() % 100) * 0.001) * MAX_X;
            addCoal(&grid, c.x(), c.y(), radius);
        }
#endif

        addLiquidCells(&grid);

        return grid;
    }

    void addLiquidCells(Grid<ContainerCell> *grid)
    {
        for (int i = 0; i < LIQUID_CELLS; ++i) {
            Coord<2> center = randCoord();
            addCell(grid, center, ZERO_INFLUXES, 0, LIQUID);
        }
    }

    void addInletOutlet(Grid<ContainerCell> *grid, bool alignVertical=false)
    {
        int spacing = 1.3 * ELEMENT_SPACING;
        int max = alignVertical? MAX_Y : MAX_X;

        for (int i = spacing; i < max; i += spacing) {
            int halfStep = spacing / 2;
            int j1 = halfStep;
            int j2 = (alignVertical? MAX_X : MAX_Y) - halfStep;
            int x1, x2, y1, y2;
            if (alignVertical) {
                x1 = j1;
                x2 = j2;
                y1 = i;
                y2 = i;
            } else {
                x1 = i;
                x2 = i;
                y1 = j1;
                y2 = j2;
            }
            addFluxCell(grid, x1, y1, INFLUXES,      0);
            addFluxCell(grid, x2, y2, ZERO_INFLUXES, EFFLUX);
        }
    }

    void addCoal(Grid<ContainerCell> *grid,
                 const int& centerX,
                 const int& centerY,
                 const int& maxRadius)
    {
        int spacing = 1.3 * ELEMENT_SPACING;

        for (double radius = 0; radius < maxRadius; radius += spacing) {
            addCircle(grid, centerX, centerY, radius, COAL);
        }
    }

    void addCircle(Grid<ContainerCell> *grid,
                   const int& centerX,
                   const int& centerY,
                   const int& radius,
                   const State& state)
    {
        const double  PI = 3.14159265;
        int spacing = 1.3 * ELEMENT_SPACING;

        // add 1 to avoid division by 0
        double circumfence = 1 + 2 * PI * radius;
        double step = spacing / circumfence * 2 * PI;
        for (double angle = 0; angle < (2 * PI); angle += step) {
            int x = centerX + cos(angle) * radius;
            int y = centerY + sin(angle) * radius;
            addCell(grid, Coord<2>(x, y), ZERO_INFLUXES, 0, state);
        }
    }

    void addNozzle(Grid<ContainerCell> *grid)
    {
        int spacing = 1.3 * ELEMENT_SPACING;

        // add narrow vertical tunnel
        for (int y = MAX_Y*0.3; y < MAX_Y*0.4; y += spacing) {
            int x = spacing;
            while(x < MAX_X*0.45) {
                addSolidCell(grid, x,         y);
                addSolidCell(grid, MAX_X - x, y);
                x += spacing;
            }
            addLiquidCell(grid, x,         y);
            addLiquidCell(grid, MAX_X - x, y);
        }

        // add horizontal barrier over the exit of the tunnel
        int x = MAX_X * 0.1;
        int y = MAX_Y * 0.6;
        addLiquidCell(grid, x,         y);
        addLiquidCell(grid, MAX_X - x, y);
        x += spacing;

        while(x < MAX_X*0.5) {
            int x1 = x;
            int x2 = MAX_X - x;

            /**
             * the barrier consists of 4 layers of cells:
             *
             *  . liquid
             *  = solid
             *  = solid
             *  . liquid
             *
             * The two solid layers ensure that no random liquid cells
             * are placed in the barrier, makin it leak. The liquid
             * cells ensure that the barrier's surface is smoooth.
             */

            int y1 = y - 2 * spacing;
            int y2 = y - 1 * spacing;
            int y3 = y + 0 * spacing;
            int y4 = y + 1 * spacing;

            addLiquidCell(grid, x1, y1);
            addLiquidCell(grid, x2, y1);

            addSolidCell(grid, x1, y2);
            addSolidCell(grid, x2, y2);
            addSolidCell(grid, x1, y3);
            addSolidCell(grid, x2, y3);

            addLiquidCell(grid, x1, y4);
            addLiquidCell(grid, x2, y4);

            x += spacing;
        }
    }

    bool checkForCollision(const Grid<ContainerCell>& grid,
                           const Coord<2>& center)
    {
        Coord<2> containerCoord = pointToContainerCoord(center);

        bool flag = true;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                const ContainerCell& container = grid[containerCoord +
                                                      Coord<2>(x, y)];
                for (int j = 0; j < container.numCells; ++j) {
                    Coord<2> delta = center - container.cells[j].center;
                    if ((delta * delta) < (ELEMENT_SPACING * ELEMENT_SPACING))
                        flag = false;
                }
            }
        }

        return flag;
    }

    void addFluxCell(Grid<ContainerCell> *grid,
                     const int& x,
                     const int& y,
                     const double influxes[SUBSTANCES],
                     const double& efflux)
    {
        addCell(grid, Coord<2>(x, y), influxes, efflux, LIQUID);
    }


    void addLiquidCell(Grid<ContainerCell> *grid,
                       const int& x,
                       const int& y)
    {
        addCell(grid, Coord<2>(x, y), ZERO_INFLUXES, 0, LIQUID);
    }

    void addSolidCell(Grid<ContainerCell> *grid,
                      const int& x,
                      const int& y)
    {
        addCell(grid, Coord<2>(x, y), ZERO_INFLUXES, 0, SOLID);
    }

    void addCell(Grid<ContainerCell> *grid,
                 const Coord<2>& center,
                 const double influxes[SUBSTANCES],
                 const double& efflux,
                 const State& state)
    {
        if (!checkForCollision(*grid, center)) {
            return;
        }

        Coord<2> containerCoord = pointToContainerCoord(center);
        int numCells = (*grid)[containerCoord].numCells;

        if (center.x() <= 0 || center.x() >= (MAX_X - 1) ||
            center.y() <= 0 || center.y() >= (MAX_Y - 1)) {
            return;
        }

        if (numCells < ContainerCell::MAX_CELLS) {
            (*grid)[containerCoord] <<
                Cell(center,
                     ID(containerCoord, numCells),
                     influxes,
                     efflux,
                     state);
        }
    }

    void fillGeometryData(Grid<ContainerCell> *grid)
    {
        int maxShape = 0;
        int maxNeighbors = 0;
        int maxCells = 0;
        double maxDiameter = 0;

        CoordBox<2> box(Coord<2>(), grid->getDimensions());
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            Coord<2> containerCoord = *iter;
            ContainerCell& container = (*grid)[containerCoord];
            maxCells = std::max(maxCells, container.numCells);

            for (int i = 0; i < container.numCells; ++i) {
                Cell& cell = container.cells[i];
                Element e(cell.center, cell.id);

                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        ContainerCell& container = (*grid)[containerCoord +
                                                           Coord<2>(x, y)];
                        for (int j = 0; j < container.numCells; ++j) {
                            if (cell.center != container.cells[j].center) {
                                e << container.cells[j];
                            }
                        }
                    }
                }

                e.updateGeometryData();
                cell.area = e.getArea();
                cell.quantities[0] = e.getArea();
                for (int i = 1; i < SUBSTANCES; ++i) {
                    cell.quantities[i] = 0;
                }

                cell.updatePressures();
                cell.setShape(e.getShape());

                for (SuperVector<Equation>::const_iterator l =
                         e.getLimits().begin();
                     l != e.getLimits().end(); ++l) {
                    if (l->neighborID.container != FarAway) {
                        cell.pushNeighbor(l->neighborID, l->length, l->dir);
                    }
                }

                maxShape     = std::max(maxShape,     cell.shapeSize);
                maxNeighbors = std::max(maxNeighbors, cell.numNeighbors);
                maxDiameter  = std::max(maxDiameter,  e.getDiameter());
            }

        }

        std::cout << "maxShape:     " << maxShape << "\n"
                  << "maxNeighbors: " << maxNeighbors << "\n"
                  << "maxCells:     " << maxCells << "\n"
                  << "maxDiameter:  " << maxDiameter << "\n\n";
    }
};

class ChromoWriter : public Writer<ContainerCell>
{
public:
    ChromoWriter(MonolithicSimulator<ContainerCell> *_sim,
                 const unsigned& _period = 1) :
        Writer<ContainerCell>("chromo", _sim, _period)
    {}

    virtual void initialized()
    {
        output(*sim->getGrid(), sim->getStep());
    }

    virtual void stepFinished()
    {
        if (sim->getStep() % period == 0) {
            output(*sim->getGrid(), sim->getStep());
        }
    }

    virtual void allDone()
    {
        output(*sim->getGrid(), sim->getStep());
    }

private:
    using Writer<ContainerCell>::sim;
    using Writer<ContainerCell>::prefix;

    int countCells(const Grid<ContainerCell>& grid)
    {
        int n = 0;
        CoordBox<2> box(Coord<2>(), grid.getDimensions());
        for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            n += grid[*i].numCells;
        }
        return n;
    }

    void writePointMesh(DBfile *dbfile, const Grid<ContainerCell>& grid,
                        const int& n)
    {
        int dim = 2;
        SuperVector<float> x;
        SuperVector<float> y;

        CoordBox<2> box(Coord<2>(), grid.getDimensions());
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell& container = grid[iter];
            for (int i = 0; i < container.numCells; ++i) {
                x << container.cells[i].center.x();
                y << container.cells[i].center.y();
            }
        }

        float *coords[] = {&x[0], &y[0]};
        DBPutPointmesh(dbfile, "centroids", dim, coords, n, DB_FLOAT, NULL);
    }

    void writeZoneMesh(DBfile *dbfile, const Grid<ContainerCell>& grid,
                       const int& n)
    {
        int dim = 2;
        SuperVector<float> x;
        SuperVector<float> y;
        SuperVector<int> nodes;
        SuperVector<int> shapeSizes;

        CoordBox<2> box(Coord<2>(), grid.getDimensions());
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell& container = grid[*iter];
            for (int i = 0; i < container.numCells; ++i) {
                SuperVector<Coord<2> > coords(container.cells[i].shapeSize);
                std::copy(container.cells[i].shape,
                          container.cells[i].shape + container.cells[i].shapeSize,
                          &coords[0]);
                shapeSizes << coords.size();
                int nodeOffset = nodes.size() + 1;
                for (int k = 0; k < coords.size(); ++k) {
                    x << coords[k].x();
                    y << coords[k].y();
                    nodes << nodeOffset + k;
                }
            }
        }

        float *coords[] = {&x[0], &y[0]};
        SuperVector<int> shapeCounts(n, 1);
        SuperVector<int> shapeTypes(n, DB_ZONETYPE_POLYGON);
        int numShapeTypes = n;
        int numNodes = x.size();
        int numZones = shapeSizes.size();
        int numListedNodes = nodes.size();

        DBPutZonelist2(dbfile, "zonelist", numZones, dim,
                       &nodes[0], numListedNodes,
                       1, 0, 0,
                       &shapeTypes[0], &shapeSizes[0], &shapeCounts[0],
                       numShapeTypes, NULL);
        DBPutUcdmesh(dbfile, "elements", dim, NULL,
                     coords, numNodes, numZones,
                     "zonelist", NULL, DB_FLOAT, NULL);
    }

    void writeSuperGrid(DBfile *dbfile, const Grid<ContainerCell>& grid,
                        const int& n)
    {
        int dim = 2;
        SuperVector<float> x;
        SuperVector<float> y;

        for (int i = 0; i < grid.getDimensions().x() + 1; ++i) {
            x << i * CELL_SPACING;
        }
        for (int i = 0; i < grid.getDimensions().y() + 1; ++i) {
            y << i * CELL_SPACING;
        }

        int dimensions[] = {x.size(), y.size()};
        float *coords[] = {&x[0], &y[0]};
        DBPutQuadmesh(dbfile, "supergrid", NULL, coords, dimensions, dim,
                      DB_FLOAT, DB_COLLINEAR, NULL);
    }

    void writeVars(DBfile *dbfile, const Grid<ContainerCell>& grid, const int& n)
    {
        CoordBox<2> box(Coord<2>(), grid.getDimensions());

#define WRITE_SCALAR(FIELD, NAME)                                       \
        {                                                               \
            SuperVector<double> buf;                                    \
            for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) { \
                const ContainerCell& container = grid[*iter];           \
                for (int i = 0; i < container.numCells; ++i)            \
                    buf << container.cells[i].FIELD;                    \
            }                                                           \
            DBPutUcdvar1(dbfile, NAME, "elements", &buf[0], n,          \
                         NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);        \
        }

        WRITE_SCALAR(area, "area");
        WRITE_SCALAR(absorbedVolume, "absorbedVolume");
        WRITE_SCALAR(pressure, "pressure");
        WRITE_SCALAR(ratios[1], "ratios_1");
        WRITE_SCALAR(ratios[2], "ratios_2");
#undef WRITE_SCALAR

        SuperVector<double> velocityX;
        SuperVector<double> velocityY;
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell& container = grid[*iter];

            for (int i = 0; i < container.numCells; ++i) {
                velocityX << container.cells[i].velocityX;
                velocityY << container.cells[i].velocityY;
            }
        }

        double *components[] = {&velocityX[0], &velocityY[0]};
        DBPutPointvar(dbfile, "velocity", "centroids", 2, components, n, DB_DOUBLE, NULL);
    }

    void output(const Grid<ContainerCell>& grid, const int& time)
    {
        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(5)
                 << time << ".silo";

        DBfile *dbfile = DBCreate(filename.str().c_str(), DB_CLOBBER, DB_LOCAL,
                                  "simulation time step", DB_HDF5);
        int n = countCells(grid);
        writePointMesh(dbfile, grid, n);
        writeZoneMesh(dbfile, grid, n);
        writeSuperGrid(dbfile, grid, n);
        writeVars(dbfile, grid, n);

        DBClose(dbfile);
    }
};


int main(int argc, char *argv[])
{
    SerialSimulator<ContainerCell> sim(
        new ChromoInitializer(
            Coord<2>(ceil(1.0 * MAX_X / CELL_SPACING),
                     ceil(1.0 * MAX_Y / CELL_SPACING)),
            300000));
    new ChromoWriter(&sim, 100);
    new TracingWriter<ContainerCell>(&sim, 100);
    sim.run();

    return 0;
}
