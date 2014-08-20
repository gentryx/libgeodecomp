#ifndef LIBGEODECOMP_IO_SILOWRITER_H
#define LIBGEODECOMP_IO_SILOWRITER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_SILO

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/storage/collectioninterface.h>
#include <libgeodecomp/storage/filterbase.h>

#include <silo.h>
#include <typeinfo>

namespace LibGeoDecomp {

namespace SiloWriterHelpers {

/**
 * This is a helper class which decouples the SiloWriter from the
 * actual type of the items for use with the Selectors.
 */
class SelectorVecBase
{
public:
    virtual ~SelectorVecBase()
    {}
};

/**
 * Dito.
 */
template<typename CELL>
class SelectorVec : public SelectorVecBase
{
public:
    typedef std::vector<Selector<CELL> > SelectorVector;
    typedef typename SelectorVector::iterator iterator;

    SelectorVec() :
        typeID(typeid(CELL))
    {}

    void addSelector(const Selector<CELL>& selector)
    {
        if (typeID != typeid(CELL)) {
            throw std::logic_error("illegal cast detected");
        }

        selectors << selector;
    }

    iterator begin()
    {
        return selectors.begin();
    }

    iterator end()
    {
        return selectors.end();
    }

private:
    const std::type_info& typeID;
    SelectorVector selectors;
};

/**
 * Dito.
 */
template<typename SILO_WRITER>
class SelectorContainer
{
public:
    typedef typename SILO_WRITER::Cell Cell;
    typedef typename SILO_WRITER::GridType GridType;

    template<typename CARGO>
    SelectorContainer(CARGO * /*unused*/) :
        selectors(new SelectorVec<CARGO>)
    {}

    virtual ~SelectorContainer()
    {
        delete selectors;
    }

    template<typename CARGO>
    void addSelector(const Selector<CARGO>& selector)
    {
        static_cast<SelectorVec<CARGO>*>(selectors)->addSelector(selector);
    }

    virtual
    void callbackAddPoints(SILO_WRITER *writer, const Cell& cell) = 0;

    virtual
    void callbackAddShapes(SILO_WRITER *writer, const Cell& cell) = 0;

    virtual
    void callbackHandleVariableForUnstructuredGrid(SILO_WRITER *writer, DBfile *dbfile, const GridType& grid) = 0;

    virtual
    void callbackHandleVariableForPointMesh(SILO_WRITER *writer, DBfile *dbfile, const GridType& grid) = 0;

protected:
    int typeId;
    SelectorVecBase *selectors;
};

/**
 * Dito.
 */
template<typename SILO_WRITER, typename COLLECTION_INTERFACE>
class SelectorContainerImplementation : public SelectorContainer<SILO_WRITER>
{
public:
    typedef typename SILO_WRITER::Cell Cell;
    typedef typename SILO_WRITER::GridType GridType;
    typedef typename COLLECTION_INTERFACE::Cargo Cargo;

    using SelectorContainer<SILO_WRITER>::selectors;

    explicit
    SelectorContainerImplementation(const COLLECTION_INTERFACE& collectionInterface) :
        SelectorContainer<SILO_WRITER>(static_cast<Cargo*>(0)),
        collectionInterface(collectionInterface)
    {}

    void callbackAddPoints(SILO_WRITER *writer, const Cell& cell)
    {
        writer->addPoints(
            collectionInterface.begin(cell),
            collectionInterface.end(cell));
    }

    void callbackAddShapes(SILO_WRITER *writer, const Cell& cell)
    {
        writer->addShapes(
            collectionInterface.begin(cell),
            collectionInterface.end(cell));
    }

    void callbackHandleVariableForUnstructuredGrid(SILO_WRITER *writer, DBfile *dbfile, const GridType& grid)
    {
        SelectorVec<Cargo> *mySelectors = static_cast<SelectorVec<Cargo>*>(selectors);
        for (typename SelectorVec<Cargo>::iterator i = mySelectors->begin(); i != mySelectors->end(); ++i) {
            writer->handleVariableForUnstructuredGrid(dbfile, grid, *i, collectionInterface);
        }
    }

    void callbackHandleVariableForPointMesh(SILO_WRITER *writer, DBfile *dbfile, const GridType& grid)
    {
        SelectorVec<Cargo> *mySelectors = static_cast<SelectorVec<Cargo>*>(selectors);
        for (typename SelectorVec<Cargo>::iterator i = mySelectors->begin(); i != mySelectors->end(); ++i) {
            writer->handleVariableForPointMesh(dbfile, grid, *i, collectionInterface);
        }
    }

private:
    COLLECTION_INTERFACE collectionInterface;
};

}

/**
 * SiloWriter makes use of the Silo library (
 * https://wci.llnl.gov/codes/silo/ ) to write regular grids,
 * unstructured grids, and point meshes, as well as variables defined
 * on those. Variables need to be defined by means of Selectors.
 *
 * Per default, all variables are scalar. If you need to write vectorial
 * data, add a selector for each vector component.
 */
template<typename CELL>
class SiloWriter : public Clonable<Writer<CELL>, SiloWriter<CELL> >
{
public:
    template<typename SILO_WRITER, typename COLLECTION_INTERFACE>
    friend class SiloWriterHelpers::SelectorContainerImplementation;

    typedef typename Writer<CELL>::GridType GridType;
    typedef typename Writer<CELL>::Topology Topology;
    typedef CELL Cell;
    typedef std::vector<Selector<Cell> > CellSelectorVec;

    using Writer<Cell>::DIM;
    using Writer<Cell>::prefix;

    /**
     * This c-tor assumes that your class provides a default interface
     * for accessing its items (e.g. particles), should there be any.
     *
     * Background: the SiloWriter needs a way to retrieve the items
     * for output from your model. This interface is described via a
     * helper class. See the unit tests of CollectionInterface for
     * further details. See the unit tests if this class to see how
     * this c-tor should be used.
     *
     * databaseType can by anything which SILO's DBCreate() accepts
     * (e.g. DB_HDF5 or DB_PDB)
     */
    SiloWriter(
        const std::string& prefix,
        const unsigned period,
        const std::string& regularGridLabel = "regular_grid",
        const std::string& unstructuredMeshLabel = "unstructured_mesh",
        const std::string& pointMeshLabel = "point_mesh",
        int databaseType = DB_PDB) :
        Clonable<Writer<CELL>, SiloWriter<CELL> >(prefix, period),
        databaseType(databaseType),
        coords(DIM),
        pointMeshSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<
            SiloWriter<CELL>,
            CollectionInterface::PassThrough<CELL> >(
                CollectionInterface::PassThrough<CELL>())),
        unstructuredGridSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<
            SiloWriter<CELL>,
            CollectionInterface::PassThrough<CELL> >(
                CollectionInterface::PassThrough<CELL>())),
        regularGridLabel(regularGridLabel),
        unstructuredMeshLabel(unstructuredMeshLabel),
        pointMeshLabel(pointMeshLabel)
    {}

    /**
     * Unlike the previous constructor, this c-tor assumes that a
     * member variable of your class holds the items for output.
     */
    template<typename CONTAINER, typename CARGO>
    SiloWriter(
        CONTAINER CARGO:: *memberPointer,
        const std::string& prefix,
        const unsigned period,
        const std::string& regularGridLabel = "regular_grid",
        const std::string& unstructuredMeshLabel = "unstructured_mesh",
        const std::string& pointMeshLabel = "point_mesh",
        int databaseType = DB_PDB) :
        Clonable<Writer<CELL>, SiloWriter<CELL> >(prefix, period),
        databaseType(databaseType),
        coords(DIM),
        pointMeshSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<SiloWriter<
            CELL>,
            CollectionInterface::Delegate<CELL, CONTAINER> >(
                CollectionInterface::Delegate<CELL, CONTAINER>(memberPointer))),
        unstructuredGridSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<
            SiloWriter<CELL>,
            CollectionInterface::Delegate<CELL, CONTAINER> >(
                CollectionInterface::Delegate<CELL, CONTAINER>(memberPointer))),
        regularGridLabel(regularGridLabel),
        unstructuredMeshLabel(unstructuredMeshLabel),
        pointMeshLabel(pointMeshLabel)
    {}

    /**
     * Same as above, but with support for two different member
     * pointers -- one for the point mesh, one for the unstructured
     * grid.
     */
    template<typename CONTAINER1, typename CONTAINER2, typename CARGO1, typename CARGO2>
    SiloWriter(
        CONTAINER1 CARGO1:: *memberPointerForPointMesh,
        CONTAINER2 CARGO2:: *memberPointerForUnstructuredGrid,
        const std::string& prefix,
        const unsigned period,
        const std::string& regularGridLabel = "regular_grid",
        const std::string& unstructuredMeshLabel = "unstructured_mesh",
        const std::string& pointMeshLabel = "point_mesh",
        int databaseType = DB_PDB) :
        Clonable<Writer<CELL>, SiloWriter<CELL> >(prefix, period),
        databaseType(databaseType),
        coords(DIM),
        pointMeshSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<SiloWriter<
            CELL>,
            CollectionInterface::Delegate<CELL, CONTAINER1> >(
                CollectionInterface::Delegate<CELL, CONTAINER1>(memberPointerForPointMesh))),
        unstructuredGridSelectors(
            new SiloWriterHelpers::SelectorContainerImplementation<
            SiloWriter<CELL>,
            CollectionInterface::Delegate<CELL, CONTAINER2> >(
                CollectionInterface::Delegate<CELL, CONTAINER2>(memberPointerForUnstructuredGrid))),
        regularGridLabel(regularGridLabel),
        unstructuredMeshLabel(unstructuredMeshLabel),
        pointMeshLabel(pointMeshLabel)
    {}

    /**
     * Adds another variable of the cargo data (e.g. the particles) to
     * this writer's output.
     */
    template<typename MEMBER, typename CARGO>
    void addSelectorForPointMesh(
        MEMBER CARGO:: *memberPointer,
        const std::string& memberName,
        const boost::shared_ptr<FilterBase<CARGO> >& filter)
    {
        addSelectorForPointMesh(Selector<CARGO>(memberPointer, memberName, filter));
    }


    template<typename MEMBER, typename CARGO>
    void addSelectorForPointMesh(
        MEMBER CARGO:: *memberPointer,
        const std::string& memberName)
    {
        addSelectorForPointMesh(Selector<CARGO>(memberPointer, memberName));
    }

    /**
     * Adds another variable of the cargo data, but associate it with
     * the unstructured grid.
     */
    template<typename MEMBER, typename CARGO>
    void addSelectorForUnstructuredGrid(
        MEMBER CARGO:: *memberPointer,
        const std::string& memberName,
        const boost::shared_ptr<FilterBase<CARGO> >& filter)
    {
        addSelectorForUnstructuredGrid(Selector<CARGO>(memberPointer, memberName, filter));
    }

    template<typename MEMBER, typename CARGO>
    void addSelectorForUnstructuredGrid(
        MEMBER CARGO:: *memberPointer,
        const std::string& memberName)
    {
        addSelectorForUnstructuredGrid(Selector<CARGO>(memberPointer, memberName));
    }

    /**
     * Adds another model variable of the cells to writer's output.
     */
    template<typename MEMBER>
    void addSelector(
        MEMBER Cell:: *memberPointer,
        const std::string& memberName,
        const boost::shared_ptr<FilterBase<Cell> >& filter)
    {
        addSelector(Selector<Cell>(memberPointer, memberName, filter));
    }

    template<typename MEMBER>
    void addSelector(
        MEMBER Cell:: *memberPointer,
        const std::string& memberName)
    {
        addSelector(Selector<Cell>(memberPointer, memberName));
    }

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(5)
                 << step << ".silo";

        DBfile *dbfile = DBCreate(filename.str().c_str(), DB_CLOBBER, DB_LOCAL,
                                  "simulation time step", databaseType);

        handleUnstructuredGrid(dbfile, grid, typename APITraits::SelectUnstructuredGrid<Cell>::Value());
        handlePointMesh(       dbfile, grid, typename APITraits::SelectPointMesh<       Cell>::Value());
        handleRegularGrid(     dbfile, grid, typename APITraits::SelectRegularGrid<     Cell>::Value());

        for (typename CellSelectorVec::iterator i = cellSelectors.begin(); i != cellSelectors.end(); ++i) {
            handleVariable(dbfile, grid, *i);
        }

        unstructuredGridSelectors->callbackHandleVariableForUnstructuredGrid(this, dbfile, grid);
        pointMeshSelectors->callbackHandleVariableForPointMesh(this, dbfile, grid);

        DBClose(dbfile);
    }

private:
    int databaseType;
    std::vector<std::vector<double> > coords;
    std::vector<int> elementTypes;
    std::vector<int> shapeSizes;
    std::vector<int> shapeCounts;
    std::vector<char> variableData;
    std::vector<int> nodeList;
    boost::shared_ptr<SiloWriterHelpers::SelectorContainer<SiloWriter<CELL> > > pointMeshSelectors;
    boost::shared_ptr<SiloWriterHelpers::SelectorContainer<SiloWriter<CELL> > > unstructuredGridSelectors;
    CellSelectorVec cellSelectors;
    Region<DIM> region;
    std::string regularGridLabel;
    std::string unstructuredMeshLabel;
    std::string pointMeshLabel;

    void addSelector(const Selector<Cell>& selector)
    {
        cellSelectors << selector;
    }

    template<typename CARGO>
    void addSelectorForPointMesh(const Selector<CARGO>& selector)
    {
        pointMeshSelectors->addSelector(selector);
    }

    template<typename CARGO>
    void addSelectorForUnstructuredGrid(const Selector<CARGO>& selector)
    {
        unstructuredGridSelectors->addSelector(selector);
    }

    void handleUnstructuredGrid(DBfile *dbfile, const GridType& grid, APITraits::TrueType)
    {
        flushDataStores();
        collectShapes(grid);
        outputUnstructuredMesh(dbfile);
    }

    void handleUnstructuredGrid(DBfile *dbfile, const GridType& grid, APITraits::FalseType)
    {
        // intentinally left blank. no need to output an unstructured grid, if the model doesn't have one.
    }

    void handlePointMesh(DBfile *dbfile, const GridType& grid, APITraits::TrueType)
    {
        flushDataStores();
        collectPoints(grid);
        outputPointMesh(dbfile);
    }

    void handlePointMesh(DBfile *dbfile, const GridType& grid, APITraits::FalseType)
    {
        // intentinally left blank. not all models have particles or such.
    }

    void handleRegularGrid(DBfile *dbfile, const GridType& grid, APITraits::TrueType)
    {
        flushDataStores();
        collectRegularGridGeometry(grid);
        outputRegularGrid(dbfile);
    }

    void handleRegularGrid(DBfile *dbfile, const GridType& grid, APITraits::FalseType)
    {
        // intentinally left blank. not all meshfree codes may want to expose this.
    }

    template<typename CARGO>
    void handleVariable(DBfile *dbfile, const GridType& grid, const Selector<CARGO>& selector)
    {
        flushDataStores();
        collectVariable(grid, selector);
        outputVariable(dbfile, selector, grid.boundingBox());
    }

    template<typename CARGO, typename COLLECTION_INTERFACE>
    void handleVariableForPointMesh(DBfile *dbfile, const GridType& grid, const Selector<CARGO>& selector, const COLLECTION_INTERFACE& collectionInterface)
    {
        flushDataStores();
        collectVariable(grid, selector, collectionInterface);
        outputVariableForPointMesh(dbfile, selector);
    }

    template<typename CARGO, typename COLLECTION_INTERFACE>
    void handleVariableForUnstructuredGrid(DBfile *dbfile, const GridType& grid, const Selector<CARGO>& selector, const COLLECTION_INTERFACE& collectionInterface)
    {
        flushDataStores();
        collectVariable(grid, selector, collectionInterface);
        outputVariableForUnstructuredGrid(dbfile, selector);
    }

    void flushDataStores()
    {
        for (int d = 0; d < DIM; ++d) {
            coords[d].resize(0);
        }

        elementTypes.resize(0);
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

            Cell cell = grid.get(*i);
            pointMeshSelectors->callbackAddPoints(this, cell);
        }
    }

    void collectShapes(const GridType& grid)
    {
        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            Cell cell = grid.get(*i);
            unstructuredGridSelectors->callbackAddShapes(this, cell);
        }
    }

    void collectVariable(const GridType& grid, const Selector<Cell>& selector)
    {
        if (region.boundingBox() != grid.boundingBox()) {
            region.clear();
            region << grid.boundingBox();
        }

        std::size_t newSize = region.size() * selector.sizeOfExternal();
        variableData.resize(newSize);
        grid.saveMemberUnchecked(&variableData[0], selector, region);
    }

    template<typename CARGO, typename COLLECTION_INTERFACE>
    void collectVariable(const GridType& grid, const Selector<CARGO>& selector, const COLLECTION_INTERFACE& collectionInterface)
    {
        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end();
             ++i) {

            Cell cell = grid.get(*i);
            std::size_t oldSize = variableData.size();
            std::size_t newSize = oldSize + collectionInterface.size(cell) * selector.sizeOfExternal();
            variableData.resize(newSize);
            addVariable(collectionInterface.begin(cell), collectionInterface.end(cell), &variableData[0] + oldSize, selector);
        }
    }

    void collectRegularGridGeometry(const GridType& grid)
    {
        Coord<DIM> dim = grid.boundingBox().dimensions;
        FloatCoord<DIM> quadrantDim;
        FloatCoord<DIM> origin;
        APITraits::SelectRegularGrid<Cell>::value(&quadrantDim, &origin);

        for (int d = 0; d < DIM; ++d) {
            for (int i = 0; i <= dim[d]; ++i) {
                coords[d] << (origin[d] + quadrantDim[d] * i);
            }
        }
    }

    void outputUnstructuredMesh(DBfile *dbfile)
    {
        DBPutZonelist2(dbfile, "zonelist", sum(shapeCounts), DIM,
                       &nodeList[0], nodeList.size(),
                       0, 0, 0,
                       &elementTypes[0], &shapeSizes[0], &shapeCounts[0],
                       elementTypes.size(), NULL);

        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        DBPutUcdmesh(
            dbfile, unstructuredMeshLabel.c_str(), DIM, NULL, tempCoords,
            nodeList.size(), sum(shapeCounts), "zonelist",
            NULL, DB_DOUBLE, NULL);

    }

    void outputPointMesh(DBfile *dbfile)
    {
        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        DBPutPointmesh(dbfile, pointMeshLabel.c_str(), DIM, tempCoords, coords[0].size(), DB_DOUBLE, NULL);
    }

    void outputRegularGrid(DBfile *dbfile)
    {
        int dimensions[DIM];
        for (int i = 0; i < DIM; ++i) {
            dimensions[i] = coords[i].size();
        }

        double *tempCoords[DIM];
        for (int d = 0; d < DIM; ++d) {
            tempCoords[d] = &coords[d][0];
        }

        DBPutQuadmesh(dbfile, regularGridLabel.c_str(), NULL, tempCoords, dimensions, DIM,
                      DB_DOUBLE, DB_COLLINEAR, NULL);
    }

    void outputVariable(DBfile *dbfile, const Selector<Cell>& selector, const CoordBox<DIM>& box)
    {
        int dimensions[DIM];
        for (int i = 0; i < DIM; ++i) {
            dimensions[i] = box.dimensions[i];
        }

        DBPutQuadvar1(
            dbfile, selector.name().c_str(), regularGridLabel.c_str(),
            &variableData[0], dimensions, DIM,
            NULL, 0, selector.siloTypeID(), DB_ZONECENT, NULL);
    }

    template<typename CARGO>
    void outputVariableForPointMesh(DBfile *dbfile, const Selector<CARGO>& selector)
    {
        DBPutPointvar1(
            dbfile, selector.name().c_str(), pointMeshLabel.c_str(),
            &variableData[0], variableData.size() / selector.sizeOfExternal(),
            selector.siloTypeID(), NULL);
    }

    template<typename CARGO>
    void outputVariableForUnstructuredGrid(DBfile *dbfile, const Selector<CARGO>& selector)
    {
        DBPutUcdvar1(
            dbfile, selector.name().c_str(), unstructuredMeshLabel.c_str(),
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

    template<typename ITERATOR, typename CARGO>
    inline void addVariable(const ITERATOR& start, const ITERATOR& end, char *target, const Selector<CARGO>& selector)
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
        elementTypes << DB_ZONETYPE_POLYGON;
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
