#ifndef LIBGEODECOMP_IO_VISITWRITER_H
#define LIBGEODECOMP_IO_VISITWRITER_H

#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/parallelization/simulator.h>
#include <libgeodecomp/storage/selector.h>

#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <cerrno>
#include <fstream>
#include <stdexcept>
#include <string>
#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>

namespace LibGeoDecomp {

namespace VisItWriterHelpers {

/**
 * This base class hides the actual member type, so it's descendants
 * can be stored more easily within the VisItWriter.
 */
template<typename CELL_TYPE>
class DataBufferBase
{
public:
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    explicit DataBufferBase(const Selector<CELL_TYPE>& selector) :
        selector(selector)
    {}

    virtual ~DataBufferBase()
    {}

    const std::string& name() const
    {
        return selector.name();
    }

    virtual visit_handle getVariable(const GridType *grid) = 0;

protected:
    Selector<CELL_TYPE> selector;
};

/**
 * This class will buffer grid data for a given member variable and can forward this data to VisIt.
 *
 * WARNING: this class is broken on some machines for members of type
 * bool, as std::vector<bool>::operator[] may not necessarily return a
 * bool& -- which is strictly necessary for us as saveMember() needs a bool*...
 */
template<typename CELL_TYPE, typename MEMBER_TYPE>
class DataBuffer : public DataBufferBase<CELL_TYPE>
{
public:
    typedef typename DataBufferBase<CELL_TYPE>::GridType GridType;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    static const int DIM = Topology::DIM;

    DataBuffer(
        const Selector<CELL_TYPE>& selector) :
        DataBufferBase<CELL_TYPE>(selector)
    {}

    virtual ~DataBuffer()
    {}

    visit_handle getVariable(const GridType *grid)
    {
        visit_handle handle = VISIT_INVALID_HANDLE;
        if (VisIt_VariableData_alloc(&handle) != VISIT_OKAY) {
            VisIt_VariableData_free(handle);
            LOG(FATAL, "Could not allocate variable buffer");
            return VISIT_INVALID_HANDLE;
        }

        CoordBox<DIM> box = grid->boundingBox();
        std::size_t expectedSize = box.dimensions.prod();
        if (dataBuffer.size() != expectedSize) {
            dataBuffer.resize(expectedSize);
            region.clear();
            region << box;
        }

        MEMBER_TYPE *p = &(dataBuffer[0]);
        grid->saveMember(p, this->selector, region);
        setData(handle, VISIT_OWNER_SIM, 1, dataBuffer.size(), &dataBuffer[0]);

        return handle;
    }

private:
    std::vector<MEMBER_TYPE> dataBuffer;
    Region<DIM> region;

    void setData(visit_handle obj, int owner, int numComponents, int numTuples, double *data)
    {
        VisIt_VariableData_setDataD(obj, owner, numComponents, numTuples, data);
    }

    void setData(visit_handle obj, int owner, int numComponents, int numTuples, int  *data)
    {
        VisIt_VariableData_setDataI(obj, owner, numComponents, numTuples, data);
    }

    void setData(visit_handle obj, int owner, int numComponents, int numTuples, float *data)
    {
        VisIt_VariableData_setDataF(obj, owner, numComponents, numTuples, data);
    }

    void setData(visit_handle obj, int owner, int numComponents, int numTuples, char *data)
    {
        VisIt_VariableData_setDataC(obj, owner, numComponents, numTuples, data);
    }

    void setData(visit_handle obj, int owner, int numComponents, int numTuples, long *data)
    {
        VisIt_VariableData_setDataL(obj, owner, numComponents, numTuples, data);
    }
};

}

/**
 * This writer uses VisIt's Libsim so users can connect to a running
 * simulation job for in situ visualization.
 */
template<typename CELL_TYPE>
class VisItWriter : public Clonable<Writer<CELL_TYPE>, VisItWriter<CELL_TYPE> >
{
public:
    typedef std::vector<boost::shared_ptr<VisItWriterHelpers::DataBufferBase<CELL_TYPE> > > DataBufferVec;
    typedef typename Writer<CELL_TYPE>::Topology Topology;
    typedef typename Writer<CELL_TYPE>::GridType GridType;
    static const int DIM = Topology::DIM;
    static const int SIMMODE_STEP = 3;


    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    /**
     * By default, a VisItWriter will create a cookie file named with
     * a time stamp and the given prefix in your home directiory (e.g.
     * $HOME/.visit/simulations/001404890671.libgeodecomp_jacobi.sim2).
     * The information in this cookie file can also be used to connect
     * from remote machines.
     *
     * The VisItWriter will remain passive and only check every period
     * time steps for inbound connections from VisIt. Set blockStart
     * if you need your application to wait for VisIt to connect
     * before starting the simulation. This is useful for debugging
     * purposes where the user wishes to inspect every time step, but
     * without having to write each and every to disk.
     */
    explicit VisItWriter(
        const std::string& prefix,
        const unsigned period = 1,
        const bool blockStart = false) :
        Clonable<Writer<CELL_TYPE>, VisItWriter<CELL_TYPE> >(prefix, period),
        blocking(0),
        runMode(blockStart? VISIT_SIMMODE_STOPPED : VISIT_SIMMODE_RUNNING)
    {
        commandsToRunModes["halt"] = VISIT_SIMMODE_STOPPED;
        commandsToRunModes["step"] = SIMMODE_STEP;
        commandsToRunModes["run" ] = VISIT_SIMMODE_RUNNING;
    }

    virtual void stepFinished(
        const GridType& newGrid, unsigned newStep, WriterEvent newEvent)
    {
        LOG(DBG, "VisItWriter::stepFinished(" << newStep << ")");

        grid = &newGrid;
        step = newStep;

        if (newEvent == WRITER_INITIALIZED) {
            return initialized();
        }

        if (newEvent == WRITER_STEP_FINISHED) {
            if (((newStep % period) != 0)) {
                return;
            }

            VisItTimeStepChanged();
            VisItUpdatePlots();
            checkVisitState();
        }

        if (newEvent == WRITER_ALL_DONE) {
            return allDone();
        }
    }

    unsigned getStep()
    {
        return step;
    }

    const GridType *getGrid()
    {
        return grid;
    }

    /**
     * Adds an accessor which allows the VisItWriter to observer another variable.
     */
    template<typename MEMBER>
    void addVariable(MEMBER CELL_TYPE:: *memberPointer, const std::string& memberName)
    {
        typedef VisItWriterHelpers::DataBuffer<CELL_TYPE, MEMBER> DataBuffer;

        Selector<CELL_TYPE> selector(memberPointer, memberName);
        DataBuffer *buffer = new DataBuffer(selector);

        variableBuffers << boost::shared_ptr<VisItWriterHelpers::DataBufferBase<CELL_TYPE> >(buffer);
    }

  private:
    std::map<std::string, int> commandsToRunModes;
    int blocking;
    int runMode;
    int dataNumber;
    DataBufferVec variableBuffers;
    const GridType *grid;
    unsigned step;

    std::vector<double> gridCoordinates[DIM];

    void initialized()
    {
        VisItSetupEnvironment();

        char buffer[1024];
        if (getcwd(buffer, sizeof(buffer)) == NULL) {
            throw std::runtime_error("no current working directory");
        }

        std::string filename = "libgeodecomp";
        if (prefix.length() > 0) {
            filename += "_" + prefix;
        }
        VisItInitializeSocketAndDumpSimFile(filename.c_str(), "", buffer, NULL, NULL, NULL);

        checkVisitState();
    }

    void allDone()
    {}

    /**
     * check if there is input from VisIt
     */
    void checkVisitState()
    {
        for (;;) {
            blocking = (runMode == VISIT_SIMMODE_RUNNING) ? 0 : 1;

            // VisItDetectInput return codes. Negative values are
            // regarded as errors:
            //
            // - -5: Logic error (fell through all cases)
            // - -4: Logic error (no descriptors but blocking)
            // - -3: Logic error (a socket was selected but not one we set)
            // - -2: Unknown error in select
            // - -1: Interrupted by EINTR in select
            // - 0: Okay - Timed out
            // - 1: Listen socket input
            // - 2: Engine socket input
            // - 3: Console socket input
            int visItState = VisItDetectInput(blocking, -1);
            LOG(DBG, "VisItDetectInput yields " << visItState);

            if (visItState <= -1) {
                LOG(FATAL, "Can't recover from error, VisIt state: " << visItState);
                throw std::runtime_error("VisItDetectInput reported error");
            }

            if (visItState == 0) {
                // There was no input from VisIt, return control to sim.
                return;
            }

            if (visItState == 1) {
                handleVisItConnection();
            }

            if (visItState == 2) {
                handleVisItInput();
            }
        }
    }

    void handleVisItConnection()
    {
        if (VisItAttemptToCompleteConnection()) {
            LOG(INFO, "VisIt connected");

            VisItSetCommandCallback(controlCommandCallback, this);
            VisItSetGetMetaData(simGetMetaData, this);
            VisItSetGetMesh(getRectilinearMesh, this);
            VisItSetGetVariable(callSetGetVariable, this);
        } else {
            char *visitError = VisItGetLastError();
            LOG(WARN, "VisIt did not connect: " << visitError);
        }
    }

    void handleVisItInput()
    {
        // VisIt wants to tell the simulation engine something:
        runMode = VISIT_SIMMODE_STOPPED;

        if (!VisItProcessEngineCommand()) {
            // Disconnect on an error or closed connection.
            VisItDisconnect();

            // Resume simulation if VisIt is gone:
            runMode = VISIT_SIMMODE_RUNNING;
        }

        if (runMode == SIMMODE_STEP) {
            runMode = VISIT_SIMMODE_STOPPED;
        }
     }

    static void controlCommandCallback(
        const char *command,
        const char *arguments,
        void *data)
    {
        LOG(INFO, "VisItWriter::controlCommandCallback(command: " << command << ", arguments: " << arguments << ")");
        VisItWriter<CELL_TYPE> *writer = static_cast<VisItWriter<CELL_TYPE>* >(data);

        writer->runMode = writer->commandsToRunModes[command];
    }


    /**
     * wrapper for callback functions needed by VisItSetGetVariable()
     */
    static visit_handle callSetGetVariable(
        int /* unused: domain*/,
        const char *name,
        void *writerHandle)
    {
        VisItWriter<CELL_TYPE> *writer = static_cast<VisItWriter<CELL_TYPE>*>(writerHandle);

        for (typename DataBufferVec::iterator i = writer->variableBuffers.begin();
             i != writer->variableBuffers.end();
             ++i) {
            if (name == (*i)->name()) {
                return (*i)->getVariable(writer->getGrid());
            }
        }

        return VISIT_INVALID_HANDLE;
    }

    /**
     * sets the meta data for VisIt: which meshes are available, and
     * which variables are defined (on which mesh)?
     */
    static visit_handle simGetMetaData(void *simData)
    {
        visit_handle handle = VISIT_INVALID_HANDLE;
        VisItWriter<CELL_TYPE> *writer = static_cast<VisItWriter<CELL_TYPE>*>(simData);

        if (VisIt_SimulationMetaData_alloc(&handle) != VISIT_OKAY) {
            return VISIT_INVALID_HANDLE;
        }

        if (writer->runMode == VISIT_SIMMODE_STOPPED) {
            VisIt_SimulationMetaData_setMode(handle, VISIT_SIMMODE_STOPPED);
        } else {
            VisIt_SimulationMetaData_setMode(handle,  VISIT_SIMMODE_RUNNING);
        }
        VisIt_SimulationMetaData_setCycleTime(handle, writer->getStep(), 0);

        handle = addMeshs(writer, handle);
        handle = addVariables(writer, handle);
        handle = addCommands(writer, handle);

        return handle;
    }

    static visit_handle addMeshs(VisItWriter<CELL_TYPE> *writer, visit_handle handle)
    {
        visit_handle meshHandle = VISIT_INVALID_HANDLE;

        // set up the mesh:
        if (VisIt_MeshMetaData_alloc(&meshHandle) != VISIT_OKAY) {
            LOG(FATAL, "Could not allocate VisIt mesh meta data");
            return VISIT_INVALID_HANDLE;
        }

        // Set the mesh's properties for 2d:
        std::string meshName = rectilinearMeshName();
        VisIt_MeshMetaData_setName(meshHandle, meshName.c_str());
        // fixme: alternative: VISIT_MESHTYPE_POINT
        VisIt_MeshMetaData_setMeshType(meshHandle, VISIT_MESHTYPE_RECTILINEAR);
        VisIt_MeshMetaData_setTopologicalDimension(meshHandle, DIM);
        VisIt_MeshMetaData_setSpatialDimension(meshHandle, DIM);
        // FIXME: do we need any units here?
        VisIt_MeshMetaData_setXUnits(meshHandle, "");
        VisIt_MeshMetaData_setXLabel(meshHandle, "Width");
        VisIt_MeshMetaData_setYUnits(meshHandle, "");
        VisIt_MeshMetaData_setYLabel(meshHandle, "Height");
        if (DIM >= 3) {
            VisIt_MeshMetaData_setZUnits(meshHandle, "");
            VisIt_MeshMetaData_setZLabel(meshHandle, "Depth");
        }
        VisIt_SimulationMetaData_addMesh(handle, meshHandle);

        return handle;
    }

    static visit_handle addVariables(VisItWriter<CELL_TYPE> *writer, visit_handle handle)
    {
        std::string meshName = rectilinearMeshName();

        for (typename DataBufferVec::iterator i = writer->variableBuffers.begin();
             i != writer->variableBuffers.end();
             ++i) {
            visit_handle variableHandle = VISIT_INVALID_HANDLE;

            if (VisIt_VariableMetaData_alloc(&variableHandle) != VISIT_OKAY) {
                LOG(FATAL, "Could not allocate VisIt variable meta data");
                return VISIT_INVALID_HANDLE;
            }

            VisIt_VariableMetaData_setName(variableHandle, (*i)->name().c_str());
            VisIt_VariableMetaData_setMeshName(variableHandle, meshName.c_str());
            VisIt_VariableMetaData_setType(variableHandle, VISIT_VARTYPE_SCALAR);
            VisIt_VariableMetaData_setCentering(variableHandle, VISIT_VARCENTERING_ZONE);

            VisIt_SimulationMetaData_addVariable(handle, variableHandle);
        }

        return handle;
    }

    static visit_handle addCommands(VisItWriter<CELL_TYPE> *writer, visit_handle handle)
    {
        for (std::map<std::string, int>::iterator i = writer->commandsToRunModes.begin();
             i != writer->commandsToRunModes.end();
             ++i) {
            visit_handle commandHandle = VISIT_INVALID_HANDLE;

            if (VisIt_CommandMetaData_alloc(&commandHandle) != VISIT_OKAY) {
                LOG(FATAL, "Cold not allocate VisIt metadata handle for command");
                return VISIT_INVALID_HANDLE;
            }

            VisIt_CommandMetaData_setName(commandHandle, i->first.c_str());
            VisIt_SimulationMetaData_addGenericCommand(handle, commandHandle);
        }

        return handle;
    }

    static std::string rectilinearMeshName()
    {
        return "mesh" + StringOps::itoa(DIM) + "d";
    }

    static visit_handle getRectilinearMesh(
        int domain,
        const char *name,
        void *connectionData)
    {
        visit_handle handle = VISIT_INVALID_HANDLE;

        VisItWriter<CELL_TYPE> *writer = static_cast<VisItWriter<CELL_TYPE>*>(connectionData);
        // add (1, 1) for 2D or (1, 1, 1) for 3D, as we're describing
        // mesh nodes here, but our data is centered on zones (nodes
        // make up the circumference of the zones).
        Coord<DIM> dims = writer->getGrid()->dimensions() + Coord<DIM>::diagonal(1);

        if (name != rectilinearMeshName()) {
            return VISIT_INVALID_HANDLE;
        }

        if (VisIt_RectilinearMesh_alloc(&handle) == VISIT_ERROR) {
            return VISIT_INVALID_HANDLE;
        }

        FloatCoord<DIM> origin;
        FloatCoord<DIM> quadrantDim;
        APITraits::SelectRegularGrid<CELL_TYPE>::value(&quadrantDim, &origin);

        for (int d = 0; d < DIM; ++d) {
            writer->gridCoordinates[d].resize(dims[d]);
            for (int i = 0; i < dims[d]; ++i) {
                writer->gridCoordinates[d][i] = origin[d] + quadrantDim[d] * i;
            }
        }

        visit_handle coordHandles[DIM];
        for (int d = 0; d < DIM; ++d) {
            VisIt_VariableData_alloc(coordHandles + d);
            VisIt_VariableData_setDataD(
                coordHandles[d],
                VISIT_OWNER_SIM,
                1,
                dims[d],
                &writer->gridCoordinates[d][0]);
        }

        if (DIM == 2) {
            VisIt_RectilinearMesh_setCoordsXY(handle, coordHandles[0], coordHandles[1]);
        }

        if (DIM == 3) {
            VisIt_RectilinearMesh_setCoordsXYZ(handle, coordHandles[0], coordHandles[1], coordHandles[2]);
        }

        return handle;
    }

    static visit_handle getPointMesh(
        int domain,
        const char *name,
        void *connectionData)
    {
        visit_handle handle = VISIT_INVALID_HANDLE;

        VisItWriter<CELL_TYPE> *writer = static_cast<VisItWriter<CELL_TYPE>*>(connectionData);

        CoordBox<DIM> box = writer->getGrid()->boundingBox();
        Coord<DIM> dims = writer->getGrid()->dimensions();
        int size = dims.prod();

        std::string expectedName = "pointmesh" + StringOps::itoa(DIM) + "d";
        if (name != expectedName) {
            return VISIT_INVALID_HANDLE;
        }

        if (VisIt_PointMesh_alloc(&handle) == VISIT_ERROR) {
            return VISIT_INVALID_HANDLE;
        }

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            Coord<DIM> coord = *i;
            int index = 0;
            // fixme: honor particle iterators here
            for (int d = 0; d < DIM; ++d) {
                writer->gridCoordinates[d][index] = coord[d];
                ++index;
            }
        }

        visit_handle coordHandles[DIM];
        for (int d = 0; d < DIM; ++d) {
            VisIt_VariableData_alloc(coordHandles + d);
            VisIt_VariableData_setDataD(
                coordHandles[d],
                VISIT_OWNER_SIM,
                1,
                size,
                &writer->gridCoordinates[d][0]);
        }

        if (DIM == 2) {
            VisIt_PointMesh_setCoordsXY(handle, coordHandles[0], coordHandles[1]);
        }

        if (DIM == 3) {
            VisIt_PointMesh_setCoordsXYZ(handle, coordHandles[0], coordHandles[1], coordHandles[2]);
        }

        return handle;
    }
};

}

#endif
