/**
 * TODO:    change C cast to C++ cast
 *          http://www.cplusplus.com/doc/tutorial/typecasting/
 * TODO:    check parameter references and make them clearer
 * TODO:    code style!
 * TODO:    exit/error Behandlung
 */

#ifndef _libgeodecomp_io_serialvisitwriter_h_
#define _libgeodecomp_io_serialvisitwriter_h_

#define SIM_STOPPED 0
#define SIMMODE_STEP 3

/*
 * VisItDetectInput RETURN CODES:
 * negative values are taken for error
 * -5: Logic error (fell through all cases)
 * -4: Logic error (no descriptors but blocking)
 * -3: Logic error (a socket was selected but not one we set)
 * -2: Unknown error in select
 * -1: Interrupted by EINTR in select
 * 0: Okay - Timed out
 * 1: Listen socket input
 * 2: Engine socket input
 * 3: Console socket input
 *
 * SerialVisitWriter errors are positive:
 */
#define UNKNOWN_TYPE 1

#include <string>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <libgeodecomp/parallelization/simulator.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/writer.h>
#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>
#include <map>
#include <boost/algorithm/string.hpp>
#include <libgeodecomp/misc/dataaccessor.h>

#include <stdio.h>

#define debugout(string) std::cout << string << std::endl;

namespace LibGeoDecomp
{

/*
 *
 */
template<typename CELL_TYPE>
class RectilinearMesh;

template < typename CELL_TYPE >
void ControlCommandCallback(const char *cmd,
                            const char *args, void *cbdata);

template < typename CELL_TYPE , typename MESH_TYPE = RectilinearMesh<CELL_TYPE> >
class SerialVisitWriter:public Writer < CELL_TYPE >
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef SerialVisitWriter<CELL_TYPE, MESH_TYPE> SVW;
    static const int DIMENSIONS = Topology::DIMENSIONS;

    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;


    /**
     * constructor
     */
    SerialVisitWriter(const std::string & _prefix,
               DataAccessor<CELL_TYPE>** _dataAccessors,
               int _numVars,
               const unsigned &_period = 1)
               : Writer < CELL_TYPE > (_prefix, _period),
               numVars(_numVars),
               dataAccessors(_dataAccessors)
    {
        runMode = 0;
        blocking = 0;
        visitstate = 0;
        error = 0;
    }

    /**
     *
     */
    void initialized()
    {
        values = new void*[numVars];

        for(int i=0; i < numVars; ++i)
        {
            variableMap[dataAccessors[i]->getName()] = i;
            initMemory(i);
            if (error != 0)
            {
                return;
            }
        }

        VisItSetupEnvironment();
        //TODO: name aus prefix lesen
        VisItInitializeSocketAndDumpSimFile("libgeodecomp", "",
                                            "", NULL, NULL, NULL);

        if (error != 0)
        {
            return;
        }

        checkVisitState();
    }

    /**
     *
     */
    void allDone()
    {
        //TODO: fix free
        //free(this->values);
    }

    /**
     *
     */
    virtual void stepFinished(
            const GridType& _grid, unsigned step, WriterEvent event)
    {
        std::cout << "finished step: " << step;
        std::cout << ", event = " << event << std::endl;

        grid = &_grid;

        if (event == WRITER_INITIALIZED)
        {
            initialized();
        }
        else if (event == WRITER_STEP_FINISHED)
        {
            VisItTimeStepChanged();

            if ((step % period) == 0)
            {
                VisItUpdatePlots();
            }

            if (error != 0)
            {
                return;
            }

            checkVisitState();
        }
        else if (event == WRITER_ALL_DONE)
        {
            allDone();
        }
    }

    /**
     *
     */
    int getRunMode()
    {
        return runMode;
    }

    /**
     *
     */
    void setRunMode(int r)
    {
        runMode = r;
    }

    /**
     *
     */
    int getNumVars()
    {
        return numVars;
    }

    /**
     *
     */
    void setError(int e)
    {
        error = e;
    }

    /**
     *
     */
    unsigned getStep()
    {
        return step;
    }

    /*
     *
     */
    const GridType* getGrid()
    {
        return grid;
    }

    /**
     *
     */
    double *x;
    double *y;
    double *z;

private:
    /**
     *
     */
    int blocking;
    int visitstate;
    int error;
    int runMode;
    int dataNumber;
    int numVars;
    DataAccessor<CELL_TYPE> **dataAccessors;
    void **values;
    std::map<std::string, int> variableMap;
    const GridType *grid;
    unsigned step;

    /**
     * get memory where simulationdata are stored
     */
    void initMemory(int i)
    {
        unsigned int size = getGrid()->boundingBox().size();

        if (strcmp("DOUBLE", dataAccessors[i]->getType().c_str()) == 0)
        {
            values[i] = new double[size];
        }
        else if (strcmp("INT", dataAccessors[i]->getType().c_str()) == 0)
        {
            values[i] = new int[size];
        }
        else if (strcmp("FLOAT", dataAccessors[i]->getType().c_str()) == 0)
        {
            values[i] = new float[size];
        }
        else if (strcmp("CHAR", dataAccessors[i]->getType().c_str()) == 0)
        {
            values[i] = new char[size];
        }
        else if (strcmp("LONG", dataAccessors[i]->getType().c_str()) == 0)
        {
            values[i] = new long[size];
        }
        else
        {
            error = UNKNOWN_TYPE;
            return;
        }
    }

    /**
     * check if there is input from visit
     */
    void checkVisitState()
    {
        do
        {
            if (error != 0)
            {
                break;
            }
            blocking = (runMode == VISIT_SIMMODE_RUNNING) ? 0 : 1;
            visitstate = VisItDetectInput(blocking, -1);
            if (visitstate <= -1)
            {
                fprintf(stderr, "Can’t recover from error!\n");
                error = visitstate;
                break;
            }
            else if (visitstate == 0)
            {
                /* There was no input from VisIt, return control to sim. */
                break;
            }
            else if (visitstate == 1)
            {
                /* VisIt is trying to connect to sim. */
                if (VisItAttemptToCompleteConnection())
                {
                    fprintf(stderr, "VisIt connected\n");
                    VisItSetCommandCallback
                            (ControlCommandCallback<CELL_TYPE>,
                                    (void *) this);
                    /* Register data access callbacks */
                    VisItSetGetMetaData(SimGetMetaData, reinterpret_cast<void*>(this));

                    typedef typename MESH_TYPE::template GetMesh<CELL_TYPE, DIMENSIONS> Mesh;
                    VisItSetGetMesh(Mesh::SimGetMesh, reinterpret_cast<void*>(this));

                    VisItSetGetVariable(wrapper, reinterpret_cast<void*>(this));
                }
                else
                {
                    char *visitError = VisItGetLastError();
                    fprintf(stderr, "VisIt did not connect: %s\n", visitError);
                }
            }
            else if (visitstate == 2)
            {
                /* VisIt wants to tell the engine something. */
                runMode = VISIT_SIMMODE_STOPPED;
                if (!VisItProcessEngineCommand())
                {
                    /* Disconnect on an error or closed connection. */
                    fprintf(stderr, "VisIt disconnected\n");
                    VisItDisconnect();
                    /* Start running again if VisIt closes. */
                    runMode = VISIT_SIMMODE_RUNNING;
                    break;
                }
                if (runMode == SIMMODE_STEP)
                {
                    runMode = VISIT_SIMMODE_STOPPED;
                    break;
                }
            }
        }
        while(true);
    }

    /**
     * wrapper for callback functions needed by
     * int VisItSetGetVariable (visit_handle(*)(int, const char *, void *) cb,
     *         void *cbdata)
     */
    static visit_handle wrapper(int domain, const char *name, void *cbdata)
    {
        typedef VisitData2<double, SetDataDouble> VisitDataDouble;
        typedef VisitData2<int, SetDataInt> VisitDataInt;
        typedef VisitData2<float, SetDataFloat> VisitDataFloat;
        typedef VisitData2<char, SetDataChar> VisitDataChar;
        typedef VisitData2<long, SetDataLong> VisitDataLong;

        SVW *sim_data = reinterpret_cast<SVW*>(cbdata);

        for(int i=0; i < sim_data->getNumVars(); ++i)
        {
            if (strcmp("DOUBLE", sim_data->dataAccessors[i]->getType().c_str()) == 0)
            {
                return VisitDataDouble::SimGetVariable(domain, name, sim_data);
            }
            else if (strcmp("INT", sim_data->dataAccessors[i]->getType().c_str()) == 0)
            {
                return VisitDataInt::SimGetVariable(domain, name, sim_data);
            }
            else if (strcmp("FLOAT", sim_data->dataAccessors[i]->getType().c_str()) == 0)
            {
                return VisitDataFloat::SimGetVariable(domain, name, sim_data);
            }
            else if (strcmp("CHAR", sim_data->dataAccessors[i]->getType().c_str()) == 0)
            {
                return VisitDataChar::SimGetVariable(domain, name, sim_data);
            }
            else if (strcmp("LONG", sim_data->dataAccessors[i]->getType().c_str()) == 0)
            {
                return VisitDataLong::SimGetVariable(domain, name, sim_data);
            }
            else
            {
                sim_data->setError(UNKNOWN_TYPE);
            }
        }
        return VISIT_INVALID_HANDLE;
    }

    /**
     * wrapper class for SetData functions
     * set the visit data for variable given by name
     */
    template < typename T, typename SETDATAFUNC>
    class VisitData2
    {
    public:
        static visit_handle SimGetVariable
                (int domain, const char *name, SVW *sim_data)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            CoordBox<DIMENSIONS> box = sim_data->getGrid()->boundingBox();
            unsigned int size = box.size();

            if(VisIt_VariableData_alloc(&h) == VISIT_OKAY)
            {
                if(sim_data->variableMap.find(name) != sim_data->variableMap.end())
                {
                    int num = sim_data->variableMap[name];
                    T *value  = (T *) sim_data->values[num];
                    unsigned j = 0;
                    for(typename CoordBox<DIMENSIONS>::Iterator i = box.begin();
                            i != box.end(); ++i)
                    {
                        sim_data->dataAccessors[num]->getFunction(
                                sim_data->getGrid()->at(*i), reinterpret_cast<void*>(&value[j]));
                        ++j;
                    }
                    SETDATAFUNC()(h, VISIT_OWNER_SIM, 1, size, value);
                }
                else
                {
                    VisIt_VariableData_free(h);
                    h = VISIT_INVALID_HANDLE;
                }
            }
            return h;
        }
    };

    /**
     *
     */
    class SetDataDouble
    {
    public:
        void operator()(visit_handle obj, int owner, int ncomps, int ntuples, double *ptr)
        {
            VisIt_VariableData_setDataD(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     *
     */
    class SetDataInt
    {
    public:
        void operator()(visit_handle obj, int owner, int ncomps, int ntuples, int  *ptr)
        {
            VisIt_VariableData_setDataI(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     *
     */
    class SetDataFloat
    {
    public:
         void operator()(visit_handle obj, int owner, int	ncomps, int ntuples, float *ptr)
        {
            VisIt_VariableData_setDataF(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     *
     */
    class SetDataChar
    {
    public:
        void operator()(visit_handle obj, int owner, int	ncomps, int ntuples, char *ptr)
        {
            VisIt_VariableData_setDataC(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     *
     */
    class SetDataLong
    {
    public:
        void operator()(visit_handle obj, int owner, int	ncomps, int ntuples, long *ptr)
        {
            VisIt_VariableData_setDataL(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     * set meta data for visit:
     *      - mesh rectilinear
     *      - varaiable type (only zonal scalar variable at the moment)
     */
    static visit_handle SimGetMetaData(void *cbdata)
    {
        visit_handle md = VISIT_INVALID_HANDLE;
        SVW *sim_data = reinterpret_cast<SVW*>(cbdata);

        if (VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
        {
            if (sim_data->runMode == VISIT_SIMMODE_STOPPED)
            {
                VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_STOPPED);
            }
            else if (sim_data->runMode == SIM_STOPPED)
            {
                VisIt_SimulationMetaData_setMode(md,  VISIT_SIMMODE_RUNNING);
            }
            VisIt_SimulationMetaData_setCycleTime(md, sim_data->getStep(), 0);

            visit_handle m1 = VISIT_INVALID_HANDLE;
            visit_handle m2 = VISIT_INVALID_HANDLE;
            visit_handle vmd = VISIT_INVALID_HANDLE;

            if (DIMENSIONS == 2)
            {
                /* Set the first mesh's properties.*/
                if (VisIt_MeshMetaData_alloc(&m1) == VISIT_OKAY)
                {
                    /* Set the mesh's properties for 2d.*/
                    VisIt_MeshMetaData_setName(m1, "mesh2d");
                    VisIt_MeshMetaData_setMeshType(m1, MESH_TYPE::getMeshType());
                    VisIt_MeshMetaData_setTopologicalDimension(m1, 0);
                    VisIt_MeshMetaData_setSpatialDimension(m1, 2);
                    VisIt_MeshMetaData_setXUnits(m1, "");
                    VisIt_MeshMetaData_setYUnits(m1, "");
                    VisIt_MeshMetaData_setXLabel(m1, "Width");
                    VisIt_MeshMetaData_setYLabel(m1, "Height");

                    VisIt_SimulationMetaData_addMesh(md, m1);
                }

                /* Add a zonal scalar variable on mesh2d. */
                for(std::map<std::string, int>::iterator it = sim_data->variableMap.begin();
                    it != sim_data->variableMap.end(); ++it)
                {
                    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
                    {
                        VisIt_VariableMetaData_setName(vmd, it->first.c_str());
                        VisIt_VariableMetaData_setMeshName(vmd, "mesh2d");
                        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

                        VisIt_SimulationMetaData_addVariable(md, vmd);
                    }
                }
            }

            if (DIMENSIONS == 3)
            {
                /* Set the second mesh's properties for 3d.*/
                if (VisIt_MeshMetaData_alloc(&m2) == VISIT_OKAY)
                {
                    /* Set the mesh's properties.*/
                    VisIt_MeshMetaData_setName(m2, "mesh3d");
                    VisIt_MeshMetaData_setMeshType(m2, MESH_TYPE::getMeshType());
                    VisIt_MeshMetaData_setTopologicalDimension(m2, 0);
                    VisIt_MeshMetaData_setSpatialDimension(m2, 3);
                    VisIt_MeshMetaData_setXUnits(m2, "");
                    VisIt_MeshMetaData_setYUnits(m2, "");
                    VisIt_MeshMetaData_setZUnits(m2, "");
                    VisIt_MeshMetaData_setXLabel(m2, "Width");
                    VisIt_MeshMetaData_setYLabel(m2, "Height");
                    VisIt_MeshMetaData_setZLabel(m2, "Depth");

                    VisIt_SimulationMetaData_addMesh(md, m2);
                }

                /* Add a zonal scalar variable on mesh3d. */
                for(std::map<std::string, int>::iterator it = sim_data->variableMap.begin();
                    it != sim_data->variableMap.end(); ++it)
                {
                    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
                    {
                        VisIt_VariableMetaData_setName(vmd, it->first.c_str());
                        VisIt_VariableMetaData_setMeshName(vmd, "mesh3d");
                        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

                        VisIt_SimulationMetaData_addVariable(md, vmd);
                    }
                }
            }
        }
        else
        {
            return VISIT_INVALID_HANDLE;
        }

        const char *cmd_names[] = { "halt", "step", "run" };
        for (int i = 0; i < sizeof(cmd_names) / sizeof(const char *);
                ++i)
        {
            visit_handle cmd = VISIT_INVALID_HANDLE;

            if (VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY)
            {
                VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
                VisIt_SimulationMetaData_addGenericCommand(md, cmd);
            }
        }
        return md;
    }

};

/*
 *
 */
template<typename CELL_TYPE>
class RectilinearMesh
{
public:
    /**
     *
     */
    static int getMeshType()
    {
        return VISIT_MESHTYPE_RECTILINEAR;
    }

    /**
     *
     */
    template<typename CELL, int DIM>
    class GetMesh;

    /**
     *
     */
    template<typename CELL>
    class GetMesh<CELL, 2>
    {
    public:
        /**
         *
         */
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            SerialVisitWriter<CELL, RectilinearMesh<CELL> >* sim_data = reinterpret_cast
                    <SerialVisitWriter<CELL, RectilinearMesh<CELL> >*>(cdata);

            int dim_x = sim_data->getGrid()->getDimensions().x() + 1;
            int dim_y = sim_data->getGrid()->getDimensions().y() + 1;

            if(strcmp(name, "mesh2d") == 0)
            {
                if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
                {
                    // TODO: free?
                    sim_data->x = new double[dim_x];
                    sim_data->y = new double[dim_y];

                    for (int i = 0; i < dim_x; ++i)
                    {
                        sim_data->x[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_y; ++i)
                    {
                        sim_data->y[i] = i*1.0;
                    }

                    visit_handle hxc, hyc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, sim_data->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, sim_data->y);
                    VisIt_RectilinearMesh_setCoordsXY(h, hxc, hyc);
                }
            }
            return h;
        }
    };

    /**
     *
     */
    template<typename CELL>
    class GetMesh<CELL, 3>
    {
    public:
        /**
         *
         */
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            SerialVisitWriter<CELL, RectilinearMesh<CELL> >* sim_data = reinterpret_cast
                    <SerialVisitWriter<CELL, RectilinearMesh<CELL> >*>(cdata);

            int dim_x = sim_data->getGrid()->getDimensions().x() + 1;
            int dim_y = sim_data->getGrid()->getDimensions().y() + 1;
            int dim_z = sim_data->getGrid()->getDimensions().z() + 1;

            if(strcmp(name, "mesh3d") == 0)
            {
                if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
                {
                    sim_data->x = new double[dim_x];
                    sim_data->y = new double[dim_y];
                    sim_data->z = new double[dim_z];

                    for (int i = 0; i < dim_x; ++i)
                    {
                        sim_data->x[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_y; ++i)
                    {
                        sim_data->y[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_z; ++i)
                    {
                        sim_data->z[i] = i*1.0;
                    }

                    visit_handle hxc, hyc, hzc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_alloc(&hzc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, sim_data->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, sim_data->y);
                    VisIt_VariableData_setDataD(hzc, VISIT_OWNER_VISIT, 1, dim_z, sim_data->z);
                    VisIt_RectilinearMesh_setCoordsXYZ(h, hxc, hyc, hzc);
                }
            }
            return h;
        }
    };
};

/*
 *
 */
template<typename CELL_TYPE>
class PointMesh
{
public:
    /**
     *
     */
    static int getMeshType()
    {
        return VISIT_MESHTYPE_RECTILINEAR;
    }

    /**
     *
     */
    template<typename CELL, int DIM>
    class GetMesh;

    /**
     *
     */
    template<typename CELL>
    class GetMesh<CELL, 2>
    {
    public:
        /**
         *
         */
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            SerialVisitWriter<CELL, PointMesh<CELL> >* sim_data = reinterpret_cast
                    <SerialVisitWriter<CELL, PointMesh<CELL> >*>(cdata);

            int dim_x = sim_data->getGrid()->getDimensions().x();
            int dim_y = sim_data->getGrid()->getDimensions().y();

            unsigned size = sim_data->getGrid()->boundingBox().size();

            if(strcmp(name, "mesh2d") == 0)
            {
                if(VisIt_PointMesh_alloc(&h) != VISIT_ERROR)
                {
                    // TODO: free?
                    sim_data->x = new double[size];
                    sim_data->y = new double[size];

                    unsigned s = 0;
                    for (int j = 0; j < dim_y; ++j)
                    {
                        for (int k = 0; k < dim_x; ++k)
                        {
                            sim_data->x[s] = k * 1.0;
                            sim_data->y[s] = j * 1.0;
                            ++s;
                        }
                    }

                    visit_handle hxc, hyc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, size, sim_data->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, size, sim_data->y);
                    VisIt_PointMesh_setCoordsXY(h, hxc, hyc);
                }
            }
            return h;
        }
    };

    /**
     *
     */
    template<typename CELL>
    class GetMesh<CELL, 3>
    {
    public:
        /**
         *
         */
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            SerialVisitWriter<CELL, PointMesh<CELL> >* sim_data = reinterpret_cast
                    <SerialVisitWriter<CELL, PointMesh<CELL> >*>(cdata);

            unsigned size = sim_data->getGrid()->boundingBox().size();

            if(strcmp(name, "mesh3d") == 0)
            {
                if(VisIt_PointMesh_alloc(&h) != VISIT_ERROR)
                {
                    sim_data->x = new double[size];
                    sim_data->y = new double[size];
                    sim_data->z = new double[size];

                    CoordBox<3> box = sim_data->getGrid()->boundingBox();

                    unsigned j = 0;
                    for(typename CoordBox<3>::Iterator i = box.begin();
                            i != box.end(); ++i)
                    {
                        sim_data->x[j] = i->x() * 1.0;
                        sim_data->y[j] = i->y() * 1.0;
                        sim_data->z[j] = i->z() * 1.0;
                        j++;
                    }

                    visit_handle hxc, hyc, hzc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_alloc(&hzc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, size, sim_data->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, size, sim_data->y);
                    VisIt_VariableData_setDataD(hzc, VISIT_OWNER_VISIT, 1, size, sim_data->z);
                    VisIt_PointMesh_setCoordsXYZ(h, hxc, hyc, hzc);
                }
            }
            return h;
        }
    };
};

/**
 *
 */
template < typename CELL_TYPE >
void ControlCommandCallback(const char *cmd,
                            const char *args, void *cbdata)
{
    SerialVisitWriter<CELL_TYPE> *sim_data = (SerialVisitWriter<CELL_TYPE> *) cbdata;

    if (strcmp(cmd, "halt") == 0)
    {
        sim_data->setRunMode(VISIT_SIMMODE_STOPPED);
    }
    else if (strcmp(cmd, "step") == 0)
    {
        //writeStep((void *) sim_data);
        sim_data->setRunMode(SIMMODE_STEP);
    }
    else if (strcmp(cmd, "run") == 0)
    {
        sim_data->setRunMode(VISIT_SIMMODE_RUNNING);
    }
}

}

#endif
