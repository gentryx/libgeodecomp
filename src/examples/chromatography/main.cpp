// fixme: refactor this demo by extracting the container cell
#include <libgeodecomp/geometry/voronoimesher.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/silowriter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <silo.h>
#include <stdlib.h>

using namespace LibGeoDecomp;

#define NOZZLE 1
#define CHROMO 2
#define SETUP 2

#if SETUP==NOZZLE
const std::size_t MAX_X   = 2000;
const std::size_t MAX_Y   = 8000;
#endif

#if SETUP==CHROMO
const std::size_t MAX_X   = 4000;
const std::size_t MAX_Y   = 10000;
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

const std::size_t SUBSTANCES = 3;
const int ELEMENT_SPACING = 50;
const int CELL_SPACING = 400;

Coord<2> FarAway(-1, -1);

class ID
{
public:
    explicit ID(const Coord<2>& containerCoord=FarAway, int index=-1) :
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

class MyCell
{
public:
    static const unsigned MAX_NEIGHBORS = 20;

    explicit MyCell(const Coord<2>& center=FarAway, const ID& _id=ID(),
         const double presetInfluxes[SUBSTANCES] = ZERO_INFLUXES,
         double efflux = 0,
         const State& state = LIQUID) :
        state(state),
        efflux(efflux),
        pressure(0),
        absorbedVolume(0),
        velocityX(0),
        velocityY(0),
        center(center),
        id(_id),
        area(0),
        numNeighbors(0)
    {
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            quantities[i] = 0;
            ratios[i] = 0;
        }

        std::copy(presetInfluxes, presetInfluxes + SUBSTANCES, influxes);
    }

    template<class CONTAINER>
    void update(CONTAINER *container, int nanoStep)
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

        for (unsigned i = 0; i < numNeighbors; ++i) {
            const MyCell& other = container->cell(neighborIDs[i]);
            if (other.state == SOLID) {
                fluxesFlow[i] = 0;
                fluxesPressure[i] = 0;
            } else {
                const FloatCoord<2>& dir = -borderDirections[i];
                double length = 1.0 / sqrt(dir * dir);
                double fluxFlow = (dir[0] * velocityX +
                                   dir[1] * velocityY) *
                    length * pressure * borderLengths[i] * FLOW_DIFFUSION;
                double fluxPressure = 0;
                if (pressure > other.pressure)
                    fluxPressure = (pressure - other.pressure) *
                        borderLengths[i] * PRESSURE_DIFFUSION;
                fluxesFlow[i]     = (std::max)(0.0, fluxFlow);
                fluxesPressure[i] = fluxPressure;
                totalFlux += fluxesPressure[i] + fluxesFlow[i];
            }
        }

        if (totalFlux > mass) {
            throw std::logic_error("unstable");
        }

        if (mass > 0) {
            double factor = (mass - totalFlux) / mass;
            for (std::size_t i = 0; i < SUBSTANCES; ++i)
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
        for (std::size_t i = 0; i < SUBSTANCES; ++i)
            fluxes[i] = 0;
        double fluxVelocityX = 0;
        double fluxVelocityY = 0;

        for (unsigned i = 0; i < numNeighbors; ++i) {
            const MyCell& other = container->cell(neighborIDs[i]);
            if (other.state != SOLID) {
                const FloatCoord<2>& dir = borderDirections[i];
                double length = 1.0 / sqrt(dir * dir);

                for (unsigned j = 0; j < other.numNeighbors; ++j) {
                    if (other.neighborIDs[j] == id) {
                        double otherMass = other.getMass();
                        if (otherMass > 0) {
                            double fluxCoefficient =
                                (other.fluxesPressure[j] + other.fluxesFlow[j]) / otherMass;
                            for (unsigned i = 0; i < SUBSTANCES; ++i)
                                fluxes[i] += fluxCoefficient * other.quantities[i];
                            fluxVelocityX += other.fluxesFlow[j] * other.velocityX;
                            fluxVelocityY += other.fluxesFlow[j] * other.velocityY;
                            double pressureCoefficient =
                                other.fluxesPressure[j] * length * PRESSURE_SPEED;
                            fluxVelocityX += dir[0] * pressureCoefficient;
                            fluxVelocityY += dir[1] * pressureCoefficient;
                        }
                        break;
                    }
                }
            }
        }

        double newMass = mass;
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            newMass += fluxes[i];
        }

        double scale = VELOCITY_MOD[state] / newMass;
        velocityX = (mass * velocityX + fluxVelocityX) * scale;
        velocityY = (mass * velocityY + fluxVelocityY) * scale;
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            quantities[i] += fluxes[i] + influxes[i];
        }

        mass = getMass();
        if (mass > 0) {
            double fluxCoefficient = 1.0 - efflux / mass;
            for (std::size_t i = 0; i < SUBSTANCES; ++i)
                quantities[i] *= fluxCoefficient;
        }

        for (std::size_t i = 0; i < SUBSTANCES; ++i)
            quantities[i] = (std::max)(0.0, quantities[i]);

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
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            substancePressures[i] = ratios[i] * pressure;
        }

        double absorbtions[SUBSTANCES];
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            absorbtions[i] = area * freeLevel * substancePressures[i] * ABSORBTION_RATES[i];
        }
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
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
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            mass += quantities[i];
        }

        return mass;
    }

    void updatePressures()
    {
        double mass = getMass();
        pressure = mass / area;
        for (std::size_t i = 0; i < SUBSTANCES; ++i) {
            ratios[i] = quantities[i] / mass;
        }
    }

    void pushNeighbor(
        const ID& id,
        double length,
        const FloatCoord<2>& dir)
    {
        if (numNeighbors >= MAX_NEIGHBORS) {
            throw std::logic_error("too many neighbors");
        }
        neighborIDs[numNeighbors] = id;
        borderLengths[numNeighbors] = length;
        borderDirections[numNeighbors] = Coord<2>(dir[0], dir[1]);
        ++numNeighbors;
    }

    std::size_t numberOfNeighbors() const
    {
        return numNeighbors;
    }

    void setShape(const std::vector<FloatCoord<2> >& newShape)
    {
        if (newShape.size() > MAX_NEIGHBORS) {
            throw std::logic_error("shape too large");
        }

        // fixme: replace these with FloatCoord?
        shape.clear();
        for (std::size_t i = 0; i < newShape.size(); ++i) {
            shape << Coord<2>(newShape[i][0], newShape[i][1]);
        }
    }

    void setArea(const double newArea)
    {
        area = newArea;
    }

    const Coord<2>& getPoint() const
    {
        return center;
    }

    const FixedArray<Coord<2>, MAX_NEIGHBORS>& getShape() const
    {
        return shape;
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
    unsigned numNeighbors;

    ID neighborIDs[MAX_NEIGHBORS];
    double borderLengths[MAX_NEIGHBORS];
    double fluxesFlow[MAX_NEIGHBORS];
    double fluxesPressure[MAX_NEIGHBORS];
    FloatCoord<2> borderDirections[MAX_NEIGHBORS];
    FixedArray<Coord<2>, MAX_NEIGHBORS> shape;
};

class ContainerCell
{
public:
    const static std::size_t MAX_CELLS = 100;
    typedef MyCell Cargo;
    typedef MyCell value_type;
    typedef MyCell* Iterator;
    typedef MyCell* iterator;
    typedef const MyCell* const_iterator;

    class API :
        public APITraits::HasCubeTopology<2>,
        public APITraits::HasNanoSteps<2>,
        public APITraits::HasIDType<ID>,
        public APITraits::HasUnstructuredGrid,
        public APITraits::HasPointMesh
    {};

    typedef Grid<ContainerCell> GridType;
    typedef CoordMap<ContainerCell, GridType> CoordMapType;

    ContainerCell() :
        numCells(0)
    {}

    const MyCell& cell(const ID id)
    {
        Coord<2> delta = id.container - coord;
        return (*neighbors)[delta].cells[id.num];
    }

    ContainerCell& operator<<(const MyCell& cell)
    {
        std::cout << "ContainerCell(" << coord << ") << " << cell.center << "\n";
        if (numCells >= MAX_CELLS) {
            throw std::logic_error("too many cells");
        }

        cells[numCells++] = cell;
        return *this;
    }

    MyCell *begin()
    {
        return cells;
    }

    const MyCell *begin() const
    {
        return cells;
    }

    MyCell *end()
    {
        return cells + numCells;
    }

    const MyCell *end() const
    {
        return cells + numCells;
    }

    void update(const CoordMapType& neighborhood, unsigned nanoStep)
    {
        neighbors = &neighborhood;
        for (std::size_t i = 0; i < numCells; ++i) {
            cells[i].update(this, nanoStep);
        }
    }

    std::size_t size() const
    {
        return numCells;
    }

    Coord<2> coord;
    MyCell cells[MAX_CELLS];
    std::size_t numCells;
    const CoordMapType *neighbors;
};

class ChromoInitializer : public SimpleInitializer<ContainerCell>, VoronoiMesher<ContainerCell>
{
public:
    ChromoInitializer(
        const Coord<2>& dim,
        unsigned steps) :
        SimpleInitializer<ContainerCell>(dim, steps),
        VoronoiMesher<ContainerCell>(dim, FloatCoord<2>(CELL_SPACING, CELL_SPACING), ELEMENT_SPACING)
    {}

    virtual void grid(GridBase<ContainerCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        ret->setEdge(ContainerCell());

        Grid<ContainerCell> grid = createBasicGrid();
        fillGeometryData(&grid);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            ContainerCell c = grid[*i];
            c.coord = *i;
            ret->set(*i, c);
        }
    }

private:

    Grid<ContainerCell> createBasicGrid()
    {
        srand(0);
        Coord<2> cellDim(ceil(1.0 * MAX_X / CELL_SPACING),
                         ceil(1.0 * MAX_Y / CELL_SPACING));
        Grid<ContainerCell> grid(cellDim);
        for (int y = 0; y < cellDim.y(); ++y) {
            for (int x = 0; x < cellDim.x(); ++x) {
                Coord<2> c(x, y);
                grid[c].coord = c;
            }
        }

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
            Coord<2> c = randomCoalCoord();
            c.y() = MAX_Y * 0.8 + c.y() * 0.1;
            double radius = (0.05 + (rand() % 100) * 0.001) * MAX_X;
            addCoal(&grid, c.x(), c.y(), radius);
        }
#endif

        addLiquidCells(&grid);

        return grid;
    }

    Coord<2> randomCoalCoord() const
    {
        int x = ELEMENT_SPACING / 2 + rand() % (MAX_X - ELEMENT_SPACING);
        int y = ELEMENT_SPACING / 2 + rand() % (MAX_Y - ELEMENT_SPACING);
        return Coord<2>(x, y);
    }

    Coord<2> pointToContainerCoord(const Coord<2>& c) const
    {
        return Coord<2>(c.x() / CELL_SPACING, c.y() / CELL_SPACING);
    }

    void addLiquidCells(Grid<ContainerCell> *grid)
    {
        CoordBox<2> box = grid->boundingBox();
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            addRandomCells(grid, *i, ContainerCell::MAX_CELLS);
        }
    }

    void addInletOutlet(Grid<ContainerCell> *grid, bool alignVertical=false)
    {
        int spacing = 1.3 * ELEMENT_SPACING;
        std::size_t max = alignVertical? MAX_Y : MAX_X;

        for (std::size_t i = spacing; i < max; i += spacing) {
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
                 int centerX,
                 int centerY,
                 int maxRadius)
    {
        int spacing = 1.3 * ELEMENT_SPACING;

        for (double radius = 0; radius < maxRadius; radius += spacing) {
            addCircle(grid, centerX, centerY, radius, COAL);
        }
    }

    void addCircle(Grid<ContainerCell> *grid,
                   int centerX,
                   int centerY,
                   int radius,
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
                for (std::size_t j = 0; j < container.numCells; ++j) {
                    Coord<2> delta = center - container.cells[j].center;
                    if ((delta * delta) < (ELEMENT_SPACING * ELEMENT_SPACING))
                        flag = false;
                }
            }
        }

        return flag;
    }

    void addFluxCell(Grid<ContainerCell> *grid,
                     int x,
                     int y,
                     const double influxes[SUBSTANCES],
                     double efflux)
    {
        addCell(grid, Coord<2>(x, y), influxes, efflux, LIQUID);
    }


    void addLiquidCell(Grid<ContainerCell> *grid,
                       int x,
                       int y)
    {
        addCell(grid, Coord<2>(x, y), ZERO_INFLUXES, 0, LIQUID);
    }

    void addSolidCell(Grid<ContainerCell> *grid,
                      int x,
                      int y)
    {
        addCell(grid, Coord<2>(x, y), ZERO_INFLUXES, 0, SOLID);
    }

    void addCell(ContainerCell *container, const FloatCoord<2>& center)
    {
        Coord<2> integerCoord(center[0], center[1]);
        (*container) << MyCell(
            integerCoord,
            ID(container->coord, container->numCells),
            ZERO_INFLUXES,
            0,
            LIQUID);
    }

    void addCell(Grid<ContainerCell> *grid,
                 const Coord<2>& center,
                 const double influxes[SUBSTANCES],
                 double efflux,
                 const State& state)
    {
        if (!checkForCollision(*grid, center)) {
            return;
        }

        Coord<2> containerCoord = pointToContainerCoord(center);
        unsigned numCells = (*grid)[containerCoord].numCells;

        if (center.x() <= 0 || center.x() >= int(MAX_X - 1) ||
            center.y() <= 0 || center.y() >= int(MAX_Y - 1)) {
            return;
        }

        if (numCells < ContainerCell::MAX_CELLS) {
            (*grid)[containerCoord] <<
                MyCell(center,
                     ID(containerCoord, numCells),
                     influxes,
                     efflux,
                     state);
        }
    }

};

class ChromoWriter : public Clonable<Writer<ContainerCell>, ChromoWriter>
{
public:
    explicit ChromoWriter(unsigned period = 1) :
        Clonable<Writer<ContainerCell>, ChromoWriter>("chromo", period)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        output(grid, step);
    }

private:
    using Writer<ContainerCell>::prefix;

    template<typename GRID_TYPE>
    int countCells(const GRID_TYPE& grid)
    {
        int n = 0;
        CoordBox<2> box = grid.boundingBox();
        for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            n += grid.get(*i).numCells;
        }
        return n;
    }

    template<typename GRID_TYPE>
    void writePointMesh(DBfile *dbfile, const GRID_TYPE& grid,
                        int n)
    {
        int dim = 2;
        std::vector<float> x;
        std::vector<float> y;

        CoordBox<2> box = grid.boundingBox();
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell container = grid.get(*iter);
            for (std::size_t i = 0; i < container.numCells; ++i) {
                x << container.cells[i].center.x();
                y << container.cells[i].center.y();
            }
        }

        float *coords[] = {&x[0], &y[0]};
        DBPutPointmesh(dbfile, "centroids", dim, coords, n, DB_FLOAT, NULL);
    }

    template<typename GRID_TYPE>
    void writeZoneMesh(DBfile *dbfile, const GRID_TYPE& grid,
                       int n)
    {
        int dim = 2;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<int> nodes;
        std::vector<int> shapeSizes;

        CoordBox<2> box = grid.boundingBox();
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell container = grid.get(*iter);
            for (std::size_t i = 0; i < container.numCells; ++i) {
                const FixedArray<Coord<2>, MyCell::MAX_NEIGHBORS>& shape = container.cells[i].shape;

                shapeSizes << shape.size();
                int nodeOffset = shape.size() + 1;
                for (std::size_t k = 0; k < shape.size(); ++k) {
                    x << shape[k].x();
                    y << shape[k].y();
                    nodes << nodeOffset + k;
                }
            }
        }

        float *coords[] = {&x[0], &y[0]};
        std::vector<int> shapeCounts(n, 1);
        std::vector<int> shapeTypes(n, DB_ZONETYPE_POLYGON);
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

    template<typename GRID_TYPE>
    void writeSuperGrid(DBfile *dbfile, const GRID_TYPE& grid,
                        int n)
    {
        int dim = 2;
        std::vector<float> x;
        std::vector<float> y;
        CoordBox<2> box = grid.boundingBox();

        for (int i = 0; i < box.dimensions.x() + 1; ++i) {
            x << i * CELL_SPACING;
        }
        for (int i = 0; i < box.dimensions.y() + 1; ++i) {
            y << i * CELL_SPACING;
        }

        int dimensions[] = {int(x.size()), int(y.size())};
        float *coords[] = {&x[0], &y[0]};
        DBPutQuadmesh(dbfile, "supergrid", NULL, coords, dimensions, dim,
                      DB_FLOAT, DB_COLLINEAR, NULL);
    }

    template<typename GRID_TYPE>
    void writeVars(DBfile *dbfile, const GRID_TYPE& grid, int n)
    {
        CoordBox<2> box = grid.boundingBox();

#define WRITE_SCALAR(FIELD, NAME)                                       \
        {                                                               \
            std::vector<double> buf;                                    \
            for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) { \
                const ContainerCell container = grid.get(*iter);        \
                for (unsigned i = 0; i < container.numCells; ++i)       \
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

        std::vector<double> velocityX;
        std::vector<double> velocityY;
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            const ContainerCell container = grid.get(*iter);

            for (unsigned i = 0; i < container.numCells; ++i) {
                velocityX << container.cells[i].velocityX;
                velocityY << container.cells[i].velocityY;
            }
        }

        double *components[] = {&velocityX[0], &velocityY[0]};
        DBPutPointvar(dbfile, "velocity", "centroids", 2, components, n, DB_DOUBLE, NULL);
    }

    template<typename GRID_TYPE>
    void output(const GRID_TYPE& grid, int time)
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
    int maxSteps = 300000;
    SerialSimulator<ContainerCell> sim(
        new ChromoInitializer(
            Coord<2>(ceil(1.0 * MAX_X / CELL_SPACING),
                     ceil(1.0 * MAX_Y / CELL_SPACING)),
            maxSteps));
    sim.addWriter(new TracingWriter<ContainerCell>(100, maxSteps));

    SiloWriter<ContainerCell> *siloWriter = new SiloWriter<ContainerCell>("chromatography", 100);

    siloWriter->addSelectorForUnstructuredGrid(&MyCell::pressure, "pressure");
    siloWriter->addSelectorForUnstructuredGrid(&MyCell::absorbedVolume, "absorbed_volume");
    // fixme: velocity
    // fixme: ratio[0]
    // fixme: ratio[1]
    // siloWriter->addSelectorForUnstructuredGrid(&MyCell::ratios, "ration0");

    sim.addWriter(siloWriter);

    sim.run();

    return 0;
}
