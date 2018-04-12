#include <libgeodecomp.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <iostream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

using namespace LibGeoDecomp;

const std::size_t MAX_NEIGHBORS = 20;

template<template<int DIM> class COORD>
class SimpleCell
{
public:
    class API :
        public APITraits::HasUnstructuredGrid,
        public APITraits::HasPointMesh,
        public APITraits::HasCustomRegularGrid
    {
    public:
        inline FloatCoord<2> getRegularGridSpacing()
        {
            return SimpleCell<COORD>::quadrantSize;
        }

        inline FloatCoord<2> getRegularGridOrigin()
        {
            return COORD<2>();
        }
    };

    explicit SimpleCell(
        const COORD<2>& center = COORD<2>(),
        const int id = 0,
        const double temperature = 0,
        const double influx = 0) :
        center(center),
        id(id),
        temperature(temperature),
        influx(influx)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        temperature = 0;

        for (FixedArray<int, MAX_NEIGHBORS>::iterator i = neighborIDs.begin();
             i != neighborIDs.end();
             ++i) {

            temperature += hood[*i].temperature;
        }

        temperature = influx + temperature / neighborIDs.size();
    }

    void setShape(const std::vector<COORD<2> >& newShape)
    {
        if (newShape.size() > MAX_NEIGHBORS) {
            std::cout << "center: " << center << "\n"
                      << "newShape: " << newShape << "\n";
            throw std::invalid_argument("shape too large");
        }

        shape.clear();
        for (typename std::vector<COORD<2> >::const_iterator i = newShape.begin();
             i != newShape.end();
             ++i) {
            shape << *i;
        }
    }

    void setArea(const double newArea)
    {
        area = newArea;
    }

    void pushNeighbor(const int neighborID, const double length, const COORD<2>& /* unused: dir */)
    {
        // ignore external boundaries, only accept other elements as
        // neighbors:
        if (neighborID == 0) {
            return;
        }

        neighborIDs << neighborID;
        neighborBorderLengths << length;
    }

    std::size_t numberOfNeighbors() const
    {
        return neighborIDs.size();
    }

    const FixedArray<COORD<2>, MAX_NEIGHBORS>& getShape() const
    {
        return shape;
    }

    const COORD<2> getPoint() const
    {
        return center;
    }

    static FloatCoord<2> quadrantSize;
    COORD<2> center;
    int id;
    double temperature;
    double influx;
    double area;
    FixedArray<COORD<2>, MAX_NEIGHBORS> shape;
    FixedArray<int, MAX_NEIGHBORS> neighborIDs;
    FixedArray<double, MAX_NEIGHBORS> neighborBorderLengths;
};

template<template<int DIM> class COORD>
FloatCoord<2> SimpleCell<COORD>::quadrantSize;

typedef ContainerCell<SimpleCell<FloatCoord>, 1000> ContainerCellType;

class VoronoiInitializer : public SimpleInitializer<ContainerCellType>,
                           public VoronoiMesher<ContainerCellType>
{
public:
    typedef ContainerCellType::Cargo Cargo;
    typedef GridBase<ContainerCellType, 2> GridType;

    VoronoiInitializer(
        const Coord<2>& dim,
        const unsigned steps,
        const std::size_t numCells,
        const double quadrantSize,
        const double elementSpacing) :
        SimpleInitializer<ContainerCellType>(dim, steps),
        VoronoiMesher<ContainerCellType>(dim, FloatCoord<2>(quadrantSize, quadrantSize), elementSpacing),
        numCells(numCells),
        counter(0)
    {
        ContainerCellType::Cargo::quadrantSize = FloatCoord<2>(quadrantSize, quadrantSize);
    }

    virtual void grid(GridType *grid)
    {
        CoordBox<2> box = grid->boundingBox();
        grid->setEdge(ContainerCellType());

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            // IDs are derived from a counter. Resetting it ensures
            // that the initializer will use the same ID for a
            // particle on each node for any given container cell:
            counter = 1 + i->toIndex(box.dimensions) * numCells;

            ContainerCellType cell = grid->get(*i);
            cell.clear();
            grid->set(*i, cell);

            if (*i == Coord<2>(0, 0)) {
                // add a single cell with an influx of 1. this will
                // make the whole system heat up over time.
                addHeater(grid, Coord<2>(0, 0), FloatCoord<2>(0, 0), 0, 1);
            }

            addRandomCells(grid, *i, numCells);
        }

        fillGeometryData(grid);
    }

private:
    std::size_t numCells;
    int counter;

    void addHeater(
        GridType *grid,
        const Coord<2>& coord,
        const FloatCoord<2>& center,
        const double temp,
        const double influx)
    {
        ContainerCellType cell = grid->get(coord);
        int id = counter++;
        cell.insert(id, SimpleCell<FloatCoord>(center, id, temp, influx));
        grid->set(coord, cell);
    }

    virtual void addCell(ContainerCellType *container, const FloatCoord<2>& center)
    {
        int id = counter++;
        // all other cells have an influx of 0:
        container->insert(id, SimpleCell<FloatCoord>(center, id, 0, 0));
    }
};

int main(int argc, char **argv)
{
    Coord<2> dim(10, 5);
    unsigned steps = 1000;
    std::size_t numCells = 100;
    double quadrantSize = 400;
    double minDistance = 100;

    SerialSimulator<ContainerCellType> sim(
        new VoronoiInitializer(dim, steps, numCells, quadrantSize, minDistance));


    SiloWriter<ContainerCellType> *siloWriter =
        new SiloWriter<ContainerCellType>("voronoi", 1);
    siloWriter->addSelectorForPointMesh(
        &SimpleCell<FloatCoord>::temperature,
        "SimpleCell_temperature");
    sim.addWriter(siloWriter);

    sim.run();

    return 0;
}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
