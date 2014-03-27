#ifndef LIBGEODECOMP_IO_SILOWRITER_H
#define LIBGEODECOMP_IO_SILOWRITER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_SILO

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/writer.h>
#include <silo.h>

namespace LibGeoDecomp {

/**
 * SiloWriter makes use of the Silo library (
 * https://wci.llnl.gov/codes/silo/ ) to write unstructured mesh data.
 */
template<typename CELL>
class SiloWriter : public Writer<CELL>
{
public:
    typedef typename Writer<CELL>::GridType GridType;
    typedef typename Writer<CELL>::Topology Topology;
    typedef typename CELL::Cargo Cargo;

    using Writer<CELL>::DIM;
    using Writer<CELL>::prefix;

    SiloWriter(
        const std::string& prefix,
        const unsigned period,
        const Selector<Cargo>& selector,
        const FloatCoord<DIM> quadrantDim) :
        Writer<CELL>(prefix, period),
        coords(DIM),
        selector(selector),
        quadrantDim(quadrantDim)
    {}

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(5)
                 << step << ".silo";

        DBfile *dbfile = DBCreate(filename.str().c_str(), DB_CLOBBER, DB_LOCAL,
                                  "simulation time step", DB_HDF5);

        flushDataStores();
        collectShapes(grid);
        outputShapeMesh(dbfile);

        flushDataStores();
        collectPoints(grid);
        outputPointMesh(dbfile);

        flushDataStores();
        collectVariable(grid);
        outputVariable(dbfile);

        flushDataStores();
        collectSuperGridGeometry(grid);
        outputSuperGrid(dbfile);

        DBClose(dbfile);
    }

private:
    std::vector<std::vector<double> > coords;
    std::vector<int> shapeTypes;
    std::vector<int> shapeSizes;
    std::vector<int> shapeCounts;
    std::vector<char> variableData;
    std::vector<int> nodeList;
    Selector<Cargo> selector;
    FloatCoord<DIM> quadrantDim;

    void flushDataStores()
    {
        for (int d = 0; d < DIM; ++d) {
            coords[d].resize(0);
        }

        shapeTypes.resize(0);
        shapeSizes.resize(0);
        shapeCounts.resize(0);
        variableData.resize(0);
        nodeList.resize(0);
    }

    void collectPoints(const GridType& grid)
    {
        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            CELL cell = grid.get(*i);
            addPoints(cell.begin(), cell.end());
        }
    }

    void collectShapes(const GridType& grid)
    {
        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            CELL cell = grid.get(*i);
            addShapes(cell.begin(), cell.end());
        }
    }

    void collectVariable(const GridType& grid)
    {
        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            CELL cell = grid.get(*i);
            std::size_t oldSize = variableData.size();
            std::size_t newSize = oldSize + cell.size() * selector.sizeOfExternal();
            variableData.resize(newSize);
            addVariable(cell.begin(), cell.end(), &variableData[0] + oldSize);
        }
    }

    void collectSuperGridGeometry(const GridType& grid)
    {
        Coord<DIM> dim = grid.boundingBox().dimensions;

        for (int d = 0; d < DIM; ++d) {
            for (int i = 0; i <= dim[d]; ++i) {
                coords[d] << quadrantDim[d] * i;
            }
        }
    }

    void outputShapeMesh(DBfile *dbfile)
    {
        DBPutZonelist2(dbfile, "zonelist", sum(shapeCounts), DIM,
                       &nodeList[0], nodeList.size(),
                       0, 0, 0,
                       &shapeTypes[0], &shapeSizes[0], &shapeCounts[0],
                       shapeTypes.size(), NULL);

        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        DBPutUcdmesh(
            dbfile, "shape_mesh", DIM, NULL, tempCoords,
            nodeList.size(), sum(shapeCounts), "zonelist",
            NULL, DB_DOUBLE, NULL);

    }

    void outputPointMesh(DBfile *dbfile)
    {
        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        // fixme: make mesh names configurable
        // fixme: make connection between variable and mesh configurable
        DBPutPointmesh(dbfile, "centroids", DIM, tempCoords, coords[0].size(), DB_DOUBLE, NULL);
    }

    void outputSuperGrid(DBfile *dbfile)
    {
        int dimensions[DIM];
        for (int i = 0; i < DIM; ++i) {
            dimensions[i] = coords[i].size();
        }

        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        DBPutQuadmesh(dbfile, "supergrid", NULL, tempCoords, dimensions, DIM,
                      DB_DOUBLE, DB_COLLINEAR, NULL);
    }

    void outputVariable(DBfile *dbfile)
    {
        DBPutUcdvar1(
            dbfile, selector.name().c_str(), "shape_mesh",
            &variableData[0], variableData.size() / selector.sizeOfExternal(),
            NULL, 0, selector.siloTypeID(), DB_ZONECENT, NULL);
    }

    inline void addCoord(const FloatCoord<1>& coord)
    {
        nodeList << coords[0].size();
        coords[0] << coord[0];
    }

    inline void addCoord(const FloatCoord<2>& coord)
    {
        nodeList << coords[0].size();
        coords[0] << coord[0];
        coords[1] << coord[1];
    }

    inline void addCoord(const FloatCoord<3>& coord)
    {
        nodeList << coords[0].size();
        coords[0] << coord[0];
        coords[1] << coord[1];
        coords[2] << coord[2];
    }

    template<typename ITERATOR>
    inline void addPoints(const ITERATOR& start, const ITERATOR& end)
    {
        for (ITERATOR i = start; i != end; ++i) {
            addPoint(i->getPoint());
        }
    }

    template<typename ITERATOR>
    inline void addShapes(const ITERATOR& start, const ITERATOR& end)
    {
        for (ITERATOR i = start; i != end; ++i) {
            addShape(i->getShape());
        }
    }

    template<typename ITERATOR>
    inline void addVariable(const ITERATOR& start, const ITERATOR& end, char *target)
    {
        char *cursor = target;

        for (ITERATOR i = start; i != end; ++i) {
            selector.copyMemberOut(&*i, cursor, 1);
            cursor += selector.sizeOfExternal();
        }
    }

    template<typename COORD_TYPE>
    inline void addPoint(const COORD_TYPE& coord)
    {
        for (int i = 0; i < DIM; ++i) {
            coords[i] << coord[i];
        }
    }

    template<typename SHAPE_CONTAINER>
    inline void addShape(const SHAPE_CONTAINER& shape)
    {
        shapeTypes << DB_ZONETYPE_POLYGON;
        shapeSizes << shape.size();
        shapeCounts << 1;

        for (typename SHAPE_CONTAINER::const_iterator i = shape.begin(); i != shape.end(); ++i) {
            addCoord(*i);
        }
    }
};

}

#endif

#endif
