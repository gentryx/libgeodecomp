#ifndef LIBGEODECOMP_IO_SILOWRITER_H
#define LIBGEODECOMP_IO_SILOWRITER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_SILO

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

    using Writer<CELL>::DIM;
    using Writer<CELL>::prefix;

    SiloWriter(
        const std::string& prefix,
        const unsigned period) :
        Writer<CELL>(prefix, period),
        coords(DIM)
    {}

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        for (int d = 0; d < DIM; ++d) {
            coords[d].resize(0);
        }
        shapeTypes.resize(0);
        shapeSizes.resize(0);
        shapeCounts.resize(0);
        nodeList.resize(0);

        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            CELL cell = grid.get(*i);
            addShapes(cell.begin(), cell.end());
        }

        output(step);
    }

    // fixme: refactor this so that a selector can extract the relevant data portions from cells
    // fixme: add functions to write variables and point-meshes
    inline void addQuad(const FloatCoord<DIM>& origin, const FloatCoord<DIM>& dimensions)
    {
        if ((shapeSizes.size() > 0) && (shapeSizes.back() == 4)) {
            shapeCounts.back() += 1;
        } else {
            shapeTypes << DB_ZONETYPE_POLYGON;
            shapeSizes << 4;
            shapeCounts << 1;
        }

        FloatCoord<DIM> upperLeft  = origin;
        FloatCoord<DIM> upperRight = origin + FloatCoord<DIM>(dimensions[0], 0);
        FloatCoord<DIM> lowerLeft  = origin + FloatCoord<DIM>(0, dimensions[1]);
        FloatCoord<DIM> lowerRight = origin + dimensions;

        addCoord(upperLeft);
        addCoord(upperRight);
        addCoord(lowerRight);
        addCoord(lowerLeft);
    }

private:
    std::vector<std::vector<double> > coords;
    std::vector<int> shapeTypes;
    std::vector<int> shapeSizes;
    std::vector<int> shapeCounts;
    std::vector<int> nodeList;

    void output(const unsigned step)
    {
        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(5)
                 << step << ".silo";

        DBfile *dbfile = DBCreate(filename.str().c_str(), DB_CLOBBER, DB_LOCAL,
                                  "simulation time step", DB_HDF5);

        DBPutZonelist2(dbfile, "zonelist", sum(shapeCounts), DIM,
                       &nodeList[0], nodeList.size(),
                       0, 0, 0,
                       &shapeTypes[0], &shapeSizes[0], &shapeCounts[0],
                       shapeTypes.size(), NULL);

        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }
        DBPutUcdmesh(dbfile, "mesh", DIM, NULL, tempCoords, nodeList.size(), sum(shapeCounts), "zonelist",
                     NULL, DB_DOUBLE, NULL);

        DBClose(dbfile);
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
    inline void addShapes(const ITERATOR& start, const ITERATOR& end)
    {
        for (ITERATOR i = start; i != end; ++i) {
            addShape(i->getShape());
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
