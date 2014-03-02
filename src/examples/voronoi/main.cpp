#include <libgeodecomp.h>
#include <iostream>

using namespace LibGeoDecomp;

// fixme
const int MAX_X = 1200;
const int MAX_Y = 1200;
const std::size_t SAMPLES = 1000;
const std::size_t NUM_CELLS = 2000;
const std::size_t MAX_NEIGHBORS = 20;
const int ELEMENT_SPACING = 10;
const int CELL_SPACING = 400;
Coord<2> FarAway(-1, -1);

// class ID
// {
// public:
//     ID(const Coord<2>& containerCoord = FarAway, const int index = -1) :
//         container(containerCoord),
//         num(index)
//     {}

//     bool operator==(const ID& other) const
//     {
//         return other.container == container && other.num == num;
//     }

//     Coord<2> container;
//     int num;
// };

typedef int ID;

Coord<2> gridDim()
{
    return Coord<2>(ceil(1.0 * MAX_X / CELL_SPACING),
                    ceil(1.0 * MAX_Y / CELL_SPACING));
}

ID makeID(Coord<2> coord, int index)
{
    // return ID(coord, index);

    // fixme: use container size rather than total number of cells
    return index + NUM_CELLS * (coord.x() + coord.y() * gridDim().x());
}

template<template<int DIM> class COORD>
class Equation
{
public:
    Equation(const COORD<2>& base, const COORD<2>& dir, ID id = ID()) :
        base(base),
        dir(dir),
        neighborID(id),
        length(-1)
    {}

    bool includes(const COORD<2>& point) const
    {
        return (point - base) * dir > 0;
    }

    COORD<2> base;
    COORD<2> dir;
    ID neighborID;
    double length;
};

template<typename _CharT, typename _Traits, template<int DIM> class COORD>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const Equation<COORD>& e)
{
    os << "Equation(base=" << e.base << ", dir" << e.dir << ")";
    return os;
}


template<template<int DIM> class COORD>
class Element
{
public:
    typedef Equation<COORD> EquationType;

    Element(const COORD<2> center = COORD<2>(1, 1), ID id = ID()) :
        center(center),
        id(id)
    {
        limits << EquationType(COORD<2>(center[0], 0),     COORD<2>(0, 1))
               << EquationType(COORD<2>(0, center[1]),     COORD<2>(1, 0))
               << EquationType(COORD<2>(MAX_X, center[1]), COORD<2>(-1, 0))
               << EquationType(COORD<2>(center[0], MAX_Y), COORD<2>(0, -1));
    }

    Element& operator<<(const EquationType& eq)
    {
        limits << eq;
        std::vector<COORD<2> > cutPoints = generateCutPoints(limits);
        // fixme: can't this be done more intelligently?
        std::set<int> deleteSet;

        for (std::size_t i = 0; i < limits. size(); ++i) {
            COORD<2> leftDir = turnLeft90(limits[i].dir);
            int dist1 = (cutPoints[2 * i + 0] - limits[i].base) * leftDir;
            int dist2 = (cutPoints[2 * i + 1] - limits[i].base) * leftDir;
            if (dist2 >= dist1) {
                // twisted differences, deleting
                deleteSet.insert(i);
            }

            for (std::size_t j = 0; j < limits.size(); ++j)
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

        std::vector<EquationType> newLimits;
        for (std::size_t i = 0; i < limits.size(); ++i) {
            if (!deleteSet.count(i)) {
                newLimits << limits[i];
            }
        }
        limits = newLimits;

        return *this;
    }

    template<typename POINT>
    Element& operator<<(const POINT& c)
    {
        COORD<2> base = (center + c.center) / 2;
        COORD<2> dir = center - c.center;
        *this << EquationType(base, dir, c.id);
        return *this;
    }

    static std::vector<COORD<2> > generateCutPoints(const std::vector<EquationType>& limits)
    {
        std::vector<COORD<2> > buf(2 * limits.size(), FarAway);

        for (std::size_t i = 0; i < limits.size(); ++i) {
            for (std::size_t j = 0; j < limits.size(); ++j) {
                if (i != j) {
                    COORD<2> cut = cutPoint(limits[i], limits[j]);
                    int offset = 2 * i;
                    COORD<2> delta = cut - limits[i].base;
                    COORD<2> turnedDir = turnLeft90(limits[i].dir);
                    double distance =
                        1.0 * delta[0] * turnedDir[0] +
                        1.0 * delta[1] * turnedDir[1];


                    bool isLeftCandidate = true;
                    if (limits[j].dir * turnedDir > 0) {
                        isLeftCandidate = false;
                        offset += 1;
                    }

                    delta = buf[offset] - limits[i].base;
                    double referenceDist =
                        1.0 * delta[0] * turnedDir[0] +
                        1.0 * delta[1] * turnedDir[1];
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

    std::vector<COORD<2> > getShape() const
    {
        std::vector<COORD<2> > cutPoints = generateCutPoints(limits);

        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            if (cutPoints[i] == FarAway) {
                throw std::logic_error("invalid cut point");
            }
        }

        std::map<double, COORD<2> > points;
        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            COORD<2> delta = cutPoints[i] - center;
            double length = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
            double dY = delta[1] / length;
            double angle = asin(dY);
            if (delta[0] < 0) {
                angle = M_PI - angle;
            }
            points[angle] = cutPoints[i];
        }

        std::vector<COORD<2> > res;
        for (typename std::map<double, COORD<2> >::iterator i = points.begin();
             i != points.end(); ++i) {
            res << i->second;
        }

        if (res.size() < 3) {
            throw std::logic_error("cycle too short");
        }

        return res;
    }

    bool includes(const COORD<2>& c)
    {
        for (std::size_t i = 0; i < limits.size(); ++i) {
            if (!limits[i].includes(c)) {
                return false;
            }
        }
        return true;
    }

    void updateGeometryData()
    {
        std::vector<COORD<2> > cutPoints = generateCutPoints(limits);

        for (std::size_t i = 0; i < limits.size(); ++i) {
            COORD<2> delta = cutPoints[2 * i + 0] - cutPoints[2 * i + 1];
            limits[i].length = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        }

        COORD<2> min(MAX_X, MAX_Y);
        COORD<2> max(0, 0);
        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            COORD<2>& c = cutPoints[i];
            max = c.max(max);
            min = c.min(min);
        }
        COORD<2> delta = max - min;

        int hits = 0;
        for (std::size_t i = 0; i < SAMPLES; ++i) {
            COORD<2> p = COORD<2>(Random::gen_d(delta[0]),
                                  Random::gen_d(delta[1])) + min;
            if (includes(p)) {
                ++hits;
            }
        }
        area = 1.0 * hits / SAMPLES * delta.prod();

        diameter = std::max(delta[0], delta[1]);
        if (diameter > CELL_SPACING/2) {
            std::cerr << "my diameter: " << diameter << "\n"
                      << "min: " << min << "\n"
                      << "max: " << max << "\n";
            throw std::logic_error("element too large");
        }
    }

    const COORD<2>& getCenter() const
    {
        return center;
    }

    const double& getArea() const
    {
        return area;
    }

    const std::vector<EquationType>& getLimits() const
    {
        return limits;
    }

    const double& getDiameter() const
    {
        return diameter;
    }

private:
    COORD<2> center;
    ID id;
    double area;
    double diameter;
    std::vector<EquationType> limits;

    static COORD<2> turnLeft90(const COORD<2>& c)
    {
        return COORD<2>(-c[1], c[0]);
    }

    static COORD<2> cutPoint(EquationType eq1, EquationType eq2)
    {
        if (eq1.dir[1] == 0) {
            if (eq2.dir[1] == 0) {
                // throw std::invalid_argument("both lines are vertical")
                return FarAway;
            }
            std::swap(eq1, eq2);
        }

        COORD<2> dir1 = turnLeft90(eq1.dir);
        double m1 = 1.0 * dir1[1] / dir1[0];
        double d1 = eq1.base[1] - m1 * eq1.base[0];

        if (eq2.dir[1] == 0) {
            return COORD<2>(eq2.base[0], eq2.base[0] * m1 + d1);
        }

        COORD<2> dir2 = turnLeft90(eq2.dir);
        double m2 = 1.0 * dir2[1] / dir2[0];
        double d2 = eq2.base[1] - m2 * eq2.base[0];

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

        return COORD<2>(x, y);
    }
};

Coord<2> randCoord()
{
    int x = ELEMENT_SPACING / 2 + Random::gen_u(MAX_X - ELEMENT_SPACING);
    int y = ELEMENT_SPACING / 2 + Random::gen_u(MAX_Y - ELEMENT_SPACING);
    return Coord<2>(x, y);
}

Coord<2> pointToContainerCoord(const Coord<2>& c)
{
    return Coord<2>(c[0] / CELL_SPACING, c[1] / CELL_SPACING);
}

template<template<int DIM> class COORD>
class SimpleCell
{
public:
    friend class Element<COORD>;
    friend class VoronoiInitializer;

    typedef Element<COORD> ElementType;
    typedef Equation<COORD> EquationType;

    SimpleCell(const COORD<2>& center = COORD<2>(), const int id = 0, const double temperature = 0) :
        center(center),
        id(id),
        temperature(temperature)
    {}

    void setShape(const std::vector<COORD<2> >& newShape)
    {
        if (newShape.size() > MAX_NEIGHBORS) {
            throw std::invalid_argument("shape too large");
        }

        shape.clear();
        for (typename std::vector<COORD<2> >::const_iterator i = newShape.begin();
             i != newShape.end();
             ++i) {
            shape << *i;
        }
    }

    void pushNeighbor(const int neighborID, const double length, const COORD<2>& /* unused: dir */)
    {
        neighbors << neighborID;
        neighborBorderLengths << length;
    }

    const FixedArray<COORD<2>, MAX_NEIGHBORS>& getShape() const
    {
        return shape;
    }

private:
    COORD<2> center;
    int id;
    double temperature;
    double area;
    FixedArray<COORD<2>, MAX_NEIGHBORS> shape;
    FixedArray<int, MAX_NEIGHBORS> neighbors;
    FixedArray<double, MAX_NEIGHBORS> neighborBorderLengths;
};

typedef ContainerCell<SimpleCell<FloatCoord>, 1000> ContainerCellType;

// fixme: refactor this demo and chromatography demo by extracting the mesh generator and container cell
class VoronoiInitializer : public SimpleInitializer<ContainerCellType>
{
public:
    typedef typename ContainerCellType::Cargo Cargo;
    typedef typename Cargo::ElementType ElementType;
    typedef typename Cargo::EquationType EquationType;

    VoronoiInitializer(
        const Coord<2>& dim,
        const unsigned& steps) :
        SimpleInitializer<ContainerCellType>(dim, steps)
    {}

    virtual void grid(GridBase<ContainerCellType, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        ret->setEdge(ContainerCellType());

        Grid<ContainerCellType> grid = createBasicGrid();
        fillGeometryData(&grid);

        for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            ContainerCellType c = grid[*i];
            ret->set(*i, c);
        }

    }

private:

    Grid<ContainerCellType> createBasicGrid()
    {
        Grid<ContainerCellType> grid(gridDim());

        for (std::size_t i = 0; i <= NUM_CELLS; ++i) {
            Coord<2> center = randCoord();
            addCell(&grid, center, 0);
        }

        return grid;
    }

    template<typename COORD_TYPE>
    double squareVector(const COORD_TYPE& vec)
    {
        return vec * vec;
    }

    bool checkForCollision(const Grid<ContainerCellType>& grid,
                           const Coord<2>& center)
    {
        Coord<2> containerCoord = pointToContainerCoord(center);

        bool flag = true;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                const ContainerCellType& container = grid[containerCoord + Coord<2>(x, y)];

                for (ContainerCellType::const_iterator j = container.begin(); j < container.end(); ++j) {
                    double length = squareVector(j->center - center);

                    if ((length * length) < (ELEMENT_SPACING * ELEMENT_SPACING)) {
                        flag = false;
                    }
                }
            }
        }

        return flag;
    }

    void addCell(Grid<ContainerCellType> *grid,
                 const Coord<2>& center,
                 double temperature)
    {
        if (!checkForCollision(*grid, center)) {
            return;
        }

        Coord<2> containerCoord = pointToContainerCoord(center);
        std::size_t numCells = (*grid)[containerCoord].size();

        if (center[0] <= 0 || center[0] >= int(MAX_X - 1) ||
            center[1] <= 0 || center[1] >= int(MAX_Y - 1)) {
            return;
        }

        if (numCells < ContainerCellType::MAX_SIZE) {
            ID id = makeID(containerCoord, numCells);
            ContainerCellType::Cargo cell(
                center,
                id,
                0);
            (*grid)[containerCoord].insert(id, cell);
        }
    }

    void fillGeometryData(Grid<ContainerCellType> *grid)
    {
        std::size_t maxShape = 0;
        std::size_t maxNeighbors = 0;
        std::size_t maxCells = 0;
        double maxDiameter = 0;

        CoordBox<2> box(Coord<2>(), grid->getDimensions());
        for (CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            Coord<2> containerCoord = *iter;
            ContainerCellType& container = (*grid)[containerCoord];
            maxCells = std::max(maxCells, container.size());

            for (ContainerCellType::Iterator i = container.begin(); i != container.end(); ++i) {
                ContainerCellType::Cargo& cell = *i;
                ElementType e(cell.center, cell.id);

                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        ContainerCellType& container =
                            (*grid)[containerCoord + Coord<2>(x, y)];
                        for (ContainerCellType::Iterator j = container.begin();
                             j != container.end();
                             ++j) {
                            if (cell.center != j->center) {
                                e << *j;
                            }
                        }
                    }
                }

                e.updateGeometryData();
                cell.area = e.getArea();
                cell.setShape(e.getShape());

                for (std::vector<EquationType>::const_iterator l =
                         e.getLimits().begin();
                     l != e.getLimits().end(); ++l) {
                    cell.pushNeighbor(l->neighborID, l->length, l->dir);
                }

                maxShape     = std::max(maxShape,     cell.shape.size());
                maxNeighbors = std::max(maxNeighbors, cell.neighbors.size());
                maxDiameter  = std::max(maxDiameter,  e.getDiameter());
            }

        }
    }
};

int main(int argc, char **argv)
{
    Coord<2> dim = gridDim();
    std::cout << "dim: " << dim << "\n";
    VoronoiInitializer init(dim, 10);
    Grid<ContainerCellType> grid(dim);
    init.grid(&grid);
    SiloWriter<ContainerCellType> writer("voronoi", 1);

    writer.stepFinished(grid, 0, WRITER_STEP_FINISHED);

    return 0;
}
