#include <libgeodecomp.h>
#include <iostream>

using namespace LibGeoDecomp;

// fixme
const int MAX_X = 800;
const int MAX_Y = 800;
const std::size_t SAMPLES = 1000;
const std::size_t NUM_CELLS = 2000;
const std::size_t MAX_NEIGHBORS = 20;
const int ELEMENT_SPACING = 10;
const int CELL_SPACING = 400;
Coord<2> FarAway(-1, -1);

class ID
{
public:
    ID(const Coord<2>& containerCoord = FarAway, const int index = -1) :
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

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const Equation& e)
{
    os << "Equation(base=" << e.base << ", dir" << e.dir << ")";
    return os;
}

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
        std::vector<Coord<2> > cutPoints = generateCutPoints(limits);
        std::set<int> deleteSet;

        for (std::size_t i = 0; i < limits. size(); ++i) {
            Coord<2> leftDir = turnLeft90(limits[i].dir);
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

        std::vector<Equation> newLimits;
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
        Coord<2> base = (center + c.center) / 2;
        Coord<2> dir = center - c.center;
        *this << Equation(base, dir, c.id);
        return *this;
    }

    static std::vector<Coord<2> > generateCutPoints(const std::vector<Equation>& limits)
    {
        std::vector<Coord<2> > buf(2 * limits.size(), FarAway);

        for (std::size_t i = 0; i < limits.size(); ++i) {
            for (std::size_t j = 0; j < limits.size(); ++j) {
                if (i != j) {
                    Coord<2> cut = cutPoint(limits[i], limits[j]);
                    int offset = 2 * i;
                    Coord<2> delta = cut - limits[i].base;
                    Coord<2> turnedDir = turnLeft90(limits[i].dir);
                    double distance =
                        1.0 * delta.x() * turnedDir.x() +
                        1.0 * delta.y() * turnedDir.y();


                    bool isLeftCandidate = true;
                    if (limits[j].dir * turnedDir > 0) {
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

    std::vector<Coord<2> > getShape() const
    {
        std::vector<Coord<2> > cutPoints = generateCutPoints(limits);

        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            if (cutPoints[i] == FarAway) {
                throw std::logic_error("invalid cut point");
            }
        }

        std::map<double, Coord<2> > points;
        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            Coord<2> delta = cutPoints[i] - center;
            double length = sqrt(delta.x() * delta.x() + delta.y() * delta.y());
            double dY = delta.y() / length;
            double angle = asin(dY);
            if (delta.x() < 0) {
                angle = M_PI - angle;
            }
            points[angle] = cutPoints[i];
        }

        std::vector<Coord<2> > res;
        for (std::map<double, Coord<2> >::iterator i = points.begin();
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
        for (std::size_t i = 0; i < limits.size(); ++i) {
            if (!limits[i].includes(c)) {
                return false;
            }
        }
        return true;
    }

    void updateGeometryData()
    {
        std::vector<Coord<2> > cutPoints = generateCutPoints(limits);

        for (std::size_t i = 0; i < limits.size(); ++i) {
            Coord<2> delta = cutPoints[2 * i + 0] - cutPoints[2 * i + 1];
            limits[i].length = sqrt(delta.x() * delta.x() + delta.y() * delta.y());
        }

        Coord<2> min(MAX_X, MAX_Y);
        Coord<2> max(0, 0);
        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            Coord<2>& c = cutPoints[i];
            max = c.max(max);
            min = c.min(min);
        }
        Coord<2> delta = max - min;

        int hits = 0;
        for (std::size_t i = 0; i < SAMPLES; ++i) {
            Coord<2> p = Coord<2>(rand() % delta.x(),
                                  rand() % delta.y()) + min;
            if (includes(p)) {
                ++hits;
            }
        }
        area = 1.0 * hits / SAMPLES * delta.prod();

        diameter = std::max(delta.x(), delta.y());
        if (diameter > CELL_SPACING/2) {
            std::cerr << "my diameter: " << diameter << "\n"
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

    const std::vector<Equation>& getLimits() const
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
    std::vector<Equation> limits;

    static Coord<2> turnLeft90(const Coord<2>& c)
    {
        return Coord<2>(-c.y(), c.x());
    }

    static Coord<2> cutPoint(Equation eq1, Equation eq2)
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

class SimpleCell
{
public:
    friend class Element;
    friend class VoronoiInitializer;

    SimpleCell(const Coord<2>& center = Coord<2>(), const ID& id = ID(), const double temperature = 0) :
        center(center),
        id(id),
        temperature(temperature)
    {}

    void setShape(const std::vector<Coord<2> >& newShape)
    {
        if (newShape.size() > MAX_NEIGHBORS) {
            throw std::invalid_argument("shape too large");
        }

        shape.clear();
        for (std::vector<Coord<2> >::const_iterator i = newShape.begin(); i != newShape.end(); ++i) {
            shape << *i;
        }
    }

    void pushNeighbor(const ID& neighborID, const double length, const Coord<2>& /* unused: dir */)
    {
        neighbors << neighborID;
        neighborBorderLengths << length;
    }

    const FixedArray<Coord<2>, MAX_NEIGHBORS>& getShape() const
    {
        return shape;
    }

private:
    Coord<2> center;
    ID id;
    double temperature;
    double area;
    FixedArray<Coord<2>, MAX_NEIGHBORS> shape;
    FixedArray<ID, MAX_NEIGHBORS> neighbors;
    FixedArray<double, MAX_NEIGHBORS> neighborBorderLengths;
};

template<std::size_t SIZE = 100>
class ContainerCell
{
public:
    friend class VoronoiInitializer;
    const static std::size_t MAX_SIZE = SIZE;

    ContainerCell(const Coord<2>& coord = Coord<2>()) :
        coord(coord)
    {}

    ContainerCell& operator<<(const SimpleCell& cell)
    {
        cells << cell;
        return *this;
    }

    std::size_t size() const
    {
        return cells.size();
    }

    typename FixedArray<SimpleCell, SIZE>::iterator begin()
    {
        return cells.begin();
    }

    typename FixedArray<SimpleCell, SIZE>::const_iterator begin() const
    {
        return cells.begin();
    }

    typename FixedArray<SimpleCell, SIZE>::iterator end()
    {
        return cells.end();
    }

    typename FixedArray<SimpleCell, SIZE>::const_iterator end() const
    {
        return cells.end();
    }

private:
    FixedArray<SimpleCell, SIZE> cells;
    Coord<2> coord;
};

typedef ContainerCell<1000> ContainerCellType;

// fixme: refactor this demo and chromatography demo by extracting the mesh generator and container cell
class VoronoiInitializer : public SimpleInitializer<ContainerCellType>
{
public:
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
            c.coord = *i;
            ret->set(*i, c);
        }
    }

private:

    Grid<ContainerCellType> createBasicGrid()
    {
        srand(0);
        Coord<2> cellDim(ceil(1.0 * MAX_X / CELL_SPACING),
                         ceil(1.0 * MAX_Y / CELL_SPACING));
        Grid<ContainerCellType> grid(cellDim);

        srand(0);
        for (std::size_t i = 0; i <= NUM_CELLS; ++i) {
            Coord<2> center = randCoord();
            addCell(&grid, center, 0);
        }

        return grid;
    }

    bool checkForCollision(const Grid<ContainerCellType>& grid,
                           const Coord<2>& center)
    {
        Coord<2> containerCoord = pointToContainerCoord(center);

        bool flag = true;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                const ContainerCellType& container = grid[containerCoord +
                                                      Coord<2>(x, y)];
                for (std::size_t j = 0; j < container.size(); ++j) {
                    Coord<2> delta = center - container.cells[j].center;
                    if ((delta * delta) < (ELEMENT_SPACING * ELEMENT_SPACING))
                        flag = false;
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
        unsigned numCells = (*grid)[containerCoord].size();

        if (center.x() <= 0 || center.x() >= int(MAX_X - 1) ||
            center.y() <= 0 || center.y() >= int(MAX_Y - 1)) {
            return;
        }

        if (numCells < ContainerCellType::MAX_SIZE) {
            (*grid)[containerCoord] <<
                SimpleCell(center,
                     ID(containerCoord, numCells),
                     0);
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

            for (std::size_t i = 0; i < container.size(); ++i) {
                SimpleCell& cell = container.cells[i];
                Element e(cell.center, cell.id);

                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        ContainerCellType& container =
                            (*grid)[containerCoord + Coord<2>(x, y)];
                        for (unsigned j = 0; j < container.size(); ++j) {
                            if (cell.center != container.cells[j].center) {
                                e << container.cells[j];
                            }
                        }
                    }
                }

                e.updateGeometryData();
                cell.area = e.getArea();
                cell.setShape(e.getShape());

                for (std::vector<Equation>::const_iterator l =
                         e.getLimits().begin();
                     l != e.getLimits().end(); ++l) {
                    if (l->neighborID.container != FarAway) {
                        cell.pushNeighbor(l->neighborID, l->length, l->dir);
                    }
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
    std::cout << "gogogo!\n";
    Coord<2> dim(ceil(1.0 * MAX_X / CELL_SPACING),
                 ceil(1.0 * MAX_Y / CELL_SPACING));
    VoronoiInitializer init(dim, 10);
    Grid<ContainerCellType> grid(dim);
    init.grid(&grid);
    SiloWriter<ContainerCellType> writer("voronoi", 1);

    writer.stepFinished(grid, 0, WRITER_STEP_FINISHED);

    return 0;
}
