#ifndef LIBGEODECOMP_GEOMETRY_VORONOIMESHER_H
#define LIBGEODECOMP_GEOMETRY_VORONOIMESHER_H

#include <libgeodecomp/geometry/convexpolytope.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/plane.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/storage/gridbase.h>
#include <algorithm>
#include <set>

namespace LibGeoDecomp {

/**
 * VoronoiMesher is a utility class which helps when setting up an
 * unstructured grid based on a Voronoi diagram. It is meant to be
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
    typedef ConvexPolytope<
        typename APITraits::SelectCoordType<CONTAINER_CELL>::Value,
        typename APITraits::SelectIDType<CONTAINER_CELL>::Value> ElementType;
    typedef Plane<
        typename APITraits::SelectCoordType<CONTAINER_CELL>::Value,
        typename APITraits::SelectIDType<CONTAINER_CELL>::Value> EquationType;
    typedef typename APITraits::SelectTopology<ContainerCellType>::Value Topology;
    static const int DIM = Topology::DIM;
    typedef GridBase<ContainerCellType, DIM> GridType;

    VoronoiMesher(const Coord<DIM>& gridDim, const FloatCoord<DIM>& quadrantSize, double minCellDistance) :
        gridDim(gridDim),
        quadrantSize(quadrantSize),
        minCellDistance(minCellDistance)
    {}

    virtual ~VoronoiMesher()
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

            if (container.size() >= numCells) {
                break;
            }

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
            maxCells = (std::max)(maxCells, container.size());

            for (typename ContainerCellType::Iterator i = container.begin(); i != container.end(); ++i) {
                Cargo& cell = *i;
                ElementType e(cell.center, simSpaceDim);
                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        ContainerCellType container2 =
                            grid->get(containerCoord + Coord<2>(x, y));
                        for (typename ContainerCellType::Iterator j = container2.begin();
                             j != container2.end();
                             ++j) {
                            if (cell.center != j->center) {
                                e << std::make_pair(j->center, i->id);
                            }
                        }
                    }
                }

                e.updateGeometryData();
                if (e.getDiameter() > quadrantSize.minElement()) {
                    throw std::logic_error("element geometry too large for container cell");
                }

                cell.setArea(e.getVolume());
                cell.setShape(e.getShape());

                for (typename std::vector<EquationType>::const_iterator l = e.getLimits().begin();
                     l != e.getLimits().end();
                     ++l) {
                    cell.pushNeighbor(l->neighborID, l->length, l->dir);
                }

                maxShape     = (std::max)(maxShape,     cell.shape.size());
                maxNeighbors = (std::max)(maxNeighbors, cell.numberOfNeighbors());
                maxDiameter  = (std::max)(maxDiameter,  e.getDiameter());
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
            ret[i] = Random::genDouble(quadrantSize[i]);
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
