#ifndef LIBGEODECOMP_GEOMETRY_VORONOIMESHER_H
#define LIBGEODECOMP_GEOMETRY_VORONOIMESHER_H

#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/storage/gridbase.h>
#include <set>

namespace LibGeoDecomp {

namespace VoronoiMesherHelpers {

template<int DIM>
Coord<DIM> farAway()
{
    return Coord<DIM>::diagonal(-1);
}

template<template<int DIM> class COORD, typename ID = int>
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

template<template<int DIM> class COORD, typename ID = int>
class Element
{
public:
    const static std::size_t SAMPLES = 1000;

    typedef Equation<COORD> EquationType;

    Element(const COORD<2>& center,
            const COORD<2>& quadrantSize,
            const COORD<2>& simSpaceDim,
            const double minCellDistance,
            ID id) :
        center(center),
        quadrantSize(quadrantSize),
        simSpaceDim(simSpaceDim),
        minCellDistance(minCellDistance),
        id(id)
    {
        limits << EquationType(COORD<2>(center[0], 0),     COORD<2>( 0,  1))
               << EquationType(COORD<2>(0, center[1]),     COORD<2>( 1,  0))
               << EquationType(COORD<2>(simSpaceDim[0], center[1]), COORD<2>(-1,  0))
               << EquationType(COORD<2>(center[0], simSpaceDim[1]), COORD<2>( 0, -1));
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
                    if (cutPoint(limits[i], limits[j]) == farAway<2>()) {
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

    std::vector<COORD<2> > generateCutPoints(const std::vector<EquationType>& limits) const
    {
        std::vector<COORD<2> > buf(2 * limits.size(), farAway<2>());

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
                    if (buf[offset] == farAway<2>()) {
                        flag = true;
                    }
                    if (isLeftCandidate  && (distance < referenceDist)) {
                        flag = true;
                    }
                    if (!isLeftCandidate && (distance > referenceDist)) {
                        flag = true;
                    }
                    if (cut == farAway<2>()) {
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
            if (cutPoints[i] == farAway<2>()) {
                throw std::logic_error("invalid cut point");
            }
        }

        std::map<double, COORD<2> > points;
        for (std::size_t i = 0; i < cutPoints.size(); ++i) {
            COORD<2> delta = cutPoints[i] - center;
            double length = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
            double angle = 0;
            if (length > 0) {
                double dY = delta[1] / length;
                angle = asin(dY);

                if (delta[0] < 0) {
                    angle = M_PI - angle;
                }
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

        COORD<2> min = simSpaceDim;
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

        double radiusSquared = delta * delta;
        double maxRadiusSquared = quadrantSize * quadrantSize * 0.5;
        if (radiusSquared > maxRadiusSquared) {
            std::cerr << "my diameter: " << diameter << "\n"
                      << "maxRadiusSquared: " << maxRadiusSquared << "\n"
                      << "quadrantSize: " << quadrantSize << "\n"
                      << "cutPoints: " << cutPoints << "\n"
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
    FloatCoord<2> quadrantSize;
    FloatCoord<2> simSpaceDim;
    double minCellDistance;
    ID id;
    double area;
    double diameter;
    std::vector<EquationType> limits;

    static COORD<2> turnLeft90(const COORD<2>& c)
    {
        return COORD<2>(-c[1], c[0]);
    }

    COORD<2> cutPoint(EquationType eq1, EquationType eq2) const
    {
        if (eq1.dir[1] == 0) {
            if (eq2.dir[1] == 0) {
                // throw std::invalid_argument("both lines are vertical")
                return farAway<2>();
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
            return farAway<2>();
        }

        double x = (d2 - d1) / (m1 - m2);
        double y = d1 + x * m1;

        if ((x < (-10 * simSpaceDim[0])) ||
            (x > ( 10 * simSpaceDim[0])) ||
            (y < (-10 * simSpaceDim[1])) ||
            (y > ( 10 * simSpaceDim[1]))) {
            return farAway<2>();
        }

        return COORD<2>(x, y);
    }
};

}

/**
 * VoronoiMesher is a utility class which helps when setting up an
 * unstructured grid based on a Voronoi diagram. It is meant to
 * embedded into an Initializer for setting up neighberhood
 * relationships, element shapes, boundary lenghts and aeras.
 *
 * It assumes that mesh generation is blocked into certain logical
 * quadrants (container cells), and that edges betreen any two cells
 * will always start/end in the same or adjacent quadrants.
 */
template<typename CONTAINER_CELL>
class VoronoiMesher
{
public:
    typedef CONTAINER_CELL ContainerCellType;
    typedef typename ContainerCellType::Cargo Cargo;
    // fixme: make discover floatcoord and int from api traits
    typedef VoronoiMesherHelpers::Element<FloatCoord, int> ElementType;
    typedef VoronoiMesherHelpers::Equation<FloatCoord, int> EquationType;
    typedef typename APITraits::SelectTopology<ContainerCellType>::Value Topology;
    static const int DIM = Topology::DIM;
    typedef GridBase<ContainerCellType, DIM> GridType;

    VoronoiMesher(const Coord<DIM>& gridDim, const FloatCoord<DIM>& quadrantSize, double minCellDistance) :
        gridDim(gridDim),
        quadrantSize(quadrantSize),
        minCellDistance(minCellDistance)
    {}

    /**
     * Adds a number of random cells to the grid cell (quadrant). The
     * seed of the random number generator will depend on the
     * quadrant's logical coordinate to ensure consistency when
     * initializing in parallel, across multiple nodes.
     */
    void addRandomCells(GridType *grid, const Coord<DIM>& coord, std::size_t numCells)
    {
        addRandomCells(grid, coord, numCells, coord.toIndex(grid->boundingBox().dimensions));
    }

    virtual void addRandomCells(GridType *grid, const Coord<DIM>& coord, std::size_t numCells, unsigned seed)
    {
        Random::seed(seed);

        for (std::size_t i = 0; i < numCells; ++i) {
            ContainerCellType container = grid->get(coord);
            FloatCoord<DIM> origin = quadrantSize.scale(coord);
            FloatCoord<DIM> location = origin + randCoord();

            insertCell(&container, location, container.begin(), container.end());
            grid->set(coord, container);
        }
    };

    void fillGeometryData(GridType *grid)
    {
        CoordBox<DIM> box = grid->boundingBox();
        FloatCoord<DIM> simSpaceDim = quadrantSize.scale(box.dimensions);
        // statistics:
        std::size_t maxShape = 0;
        std::size_t maxNeighbors = 0;
        std::size_t maxCells = 0;
        double maxDiameter = 0;

        for (typename CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            Coord<2> containerCoord = *iter;
            ContainerCellType container = grid->get(containerCoord);
            maxCells = std::max(maxCells, container.size());

            for (typename ContainerCellType::Iterator i = container.begin(); i != container.end(); ++i) {
                Cargo& cell = *i;
                ElementType e(cell.center, quadrantSize, simSpaceDim, minCellDistance, cell.id);

                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        ContainerCellType container =
                            grid->get(containerCoord + Coord<2>(x, y));
                        for (typename ContainerCellType::Iterator j = container.begin();
                             j != container.end();
                             ++j) {
                            if (cell.center != j->center) {
                                e << *j;
                            }
                        }
                    }
                }

                e.updateGeometryData();
                cell.setArea(e.getArea());
                cell.setShape(e.getShape());

                for (std::vector<EquationType>::const_iterator l = e.getLimits().begin();
                     l != e.getLimits().end();
                     ++l) {
                    cell.pushNeighbor(l->neighborID, l->length, l->dir);
                }

                maxShape     = std::max(maxShape,     cell.shape.size());
                maxNeighbors = std::max(maxNeighbors, cell.neighborIDs.size());
                maxDiameter  = std::max(maxDiameter,  e.getDiameter());
            }

            grid->set(containerCoord, container);
        }

        LOG(DBG,
            "VoronoiMesher::fillGeometryData(maxShape: " << maxShape
            << ", maxNeighbors: " << maxNeighbors
            << ", maxDiameter: " << maxDiameter
            << ", maxCells: " << maxCells << ")");
    }

    virtual void addCell(ContainerCellType *container, const FloatCoord<DIM>& center) = 0;

protected:
    Coord<DIM> gridDim;
    FloatCoord<DIM> quadrantSize;
    double minCellDistance;


    FloatCoord<DIM> randCoord()
    {
        FloatCoord<DIM> ret;
        for (int i = 0; i < DIM; ++i) {
            ret[i] = Random::gen_d(quadrantSize[i]);
        }

        return ret;
    }

    template<typename ITERATOR>
    void insertCell(
        ContainerCellType *container,
        const FloatCoord<DIM>& center,
        const ITERATOR& begin,
        const ITERATOR& end)
    {
        for (ITERATOR i = begin; i != end; ++i) {
            FloatCoord<DIM> delta = center - i->center;
            double distanceSquared = delta * delta;
            if (distanceSquared < minCellDistance) {
                return;
            }
        }

        addCell(container, center);
    }

};

}


#endif
