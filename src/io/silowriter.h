#ifndef LIBGEODECOMP_IO_SILOWRITER_H
#define LIBGEODECOMP_IO_SILOWRITER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_SILO

#include <libgeodecomp/io/writer.h>
#include <silo.h>

namespace LibGeoDecomp {

namespace SiloWriterHelpers {

template<typename CONTAINER_CELL, typename CELL, typename MEMBER>
class SelectorContainer
{
};

}

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

    using Writer<CELL>::DIM;
    using Writer<CELL>::prefix;

    template<typename CARGO_CELL>
    SiloWriter(
        const std::string& prefix,
        const unsigned period,
        // fixme: actually use the selector, maybe even a vector of selectors?
        const Selector<CARGO_CELL>& selector,
        const FloatCoord<DIM> quadrantDim) :
        Writer<CELL>(prefix, period),
        coords(DIM),
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

    // fixme: add functions to write variables and point-meshes

private:
    std::vector<std::vector<double> > coords;
    // fixme: deduce type from selector
    std::vector<double> variableData;
    std::vector<int> shapeTypes;
    std::vector<int> shapeSizes;
    std::vector<int> shapeCounts;
    std::vector<int> nodeList;
    // Selector<CELL> selector;
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
            addVariable(cell.begin(), cell.end());
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
        DBPutUcdvar1(dbfile, "fixme_varname", "shape_mesh", &variableData[0], variableData.size(),
                     NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
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
    inline void addVariable(const ITERATOR& start, const ITERATOR& end)
    {
        for (ITERATOR i = start; i != end; ++i) {
            double data;
            // fixme
            // selector(&*i, &data, 1);
            data = i->temperature;
            variableData << data;
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
