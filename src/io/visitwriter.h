#ifndef LIBGEODECOMP_IO_VISITWRITER_H
#define LIBGEODECOMP_IO_VISITWRITER_H

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
 */

#define UNKNOWN_TYPE 1
#define NO_CWD 2

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
#include <unistd.h>

#include <stdio.h>

#define debugout(string) std::cout << string << std::endl;

namespace LibGeoDecomp {

class RectilinearMesh;

template<typename CELL_TYPE>
void ControlCommandCallback(const char *cmd,
                            const char *args, void *cbdata);

template<typename CELL_TYPE , typename MESH_TYPE=RectilinearMesh>
class VisitWriter : public Writer<CELL_TYPE>
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef VisitWriter<CELL_TYPE, MESH_TYPE> SVW;
    static const int DIMENSIONS = Topology::DIM;

    using Writer<CELL_TYPE>::period;

    // fixme: rename to VisItWriter
    // fixme: needs test
    VisitWriter(
        DataAccessor<CELL_TYPE> **dataAccessors,
        int numVars,
        const unsigned& period=1,
        const int& runMode=0) :
        Writer<CELL_TYPE>("", period),
        runMode(runMode),
        numVars(numVars),
        dataAccessors(dataAccessors)
    {
        // fixme: do this in initializer list
        blocking = 0;
        visitState = 0;
        error = 0;
    }

    // fixme: parameter names shouldn't start with underscore
    virtual void stepFinished(
        const GridType& _grid, unsigned _step, WriterEvent _event)
    {
        grid = &_grid;
        step = _step;

        if (_event == WRITER_INITIALIZED) {
            initialized();
        } else if (_event == WRITER_STEP_FINISHED) {
            VisItTimeStepChanged();

            if (((_step % period) == 0) && (VisItIsConnected())) {
                std::cout << "VisItWriter::stepFinished()" << std::endl;
                std::cout << "  step: " << _step << std::endl;
                std::cout << "  event: " << _event << std::endl;
                if (VisItIsConnected()) {
                    VisItUpdatePlots();
                }
            }

            if (error != 0) {
                return;
            }

            checkVisitState();
        } else if (_event == WRITER_ALL_DONE) {
            allDone();
        }
    }

    int getRunMode()
    {
        return runMode;
    }

    void setRunMode(int r)
    {
        runMode = r;
    }

    int getNumVars()
    {
        return numVars;
    }

    void setError(int e)
    {
        error = e;
    }

    unsigned getStep()
    {
        return step;
    }

    const GridType *getGrid()
    {
        return grid;
    }

    // fixme: why not private?
    double *x;
    double *y;
    double *z;

  private:
    int blocking;
    int visitState;
    int error;
    int runMode;
    int dataNumber;
    int numVars;
    DataAccessor<CELL_TYPE> **dataAccessors;
    void **values;
    std::map<std::string, int> variableMap;
    const GridType *grid;
    unsigned step;

    void deleteVarMem(int i)
    {
        if (strcmp("DOUBLE", dataAccessors[i]->getType().c_str()) == 0) {
            delete [] (reinterpret_cast<double*>(values[i]));
        } else if (strcmp("INT", dataAccessors[i]->getType().c_str()) == 0) {
            delete [] (reinterpret_cast<int*>(values[i]));
        } else if (strcmp("FLOAT", dataAccessors[i]->getType().c_str()) == 0) {
            delete [] (reinterpret_cast<float*>(values[i]));
        } else if (strcmp("CHAR", dataAccessors[i]->getType().c_str()) == 0) {
            delete [] (reinterpret_cast<char*>(values[i]));
        } else if (strcmp("LONG", dataAccessors[i]->getType().c_str()) == 0) {
            delete [] (reinterpret_cast<long*>(values[i]));
        }
    }

    /*
     * get memory where variabledata are stored
     */
    void initVarMem(int i)
    {
        unsigned int size = getGrid()->boundingBox().size();

        if (strcmp("DOUBLE", dataAccessors[i]->getType().c_str()) == 0) {
            values[i] = new double[size];
        } else if (strcmp("INT", dataAccessors[i]->getType().c_str()) == 0) {
            values[i] = new int[size];
        } else if (strcmp("FLOAT", dataAccessors[i]->getType().c_str()) == 0) {
            values[i] = new float[size];
        } else if (strcmp("CHAR", dataAccessors[i]->getType().c_str()) == 0) {
            values[i] = new char[size];
        } else if (strcmp("LONG", dataAccessors[i]->getType().c_str()) == 0) {
            values[i] = new long[size];
        } else {
            error = UNKNOWN_TYPE;
            return;
        }
    }

    /*
     * initialize memory for all variables
     */
    void initMemory()
    {
        values = new void*[numVars];

        for(int i=0; i < numVars; ++i) {
            variableMap[dataAccessors[i]->getName()] = i;
            initVarMem(i);
            if (error != 0) {
                return;
            }
        }
    }

    /*
     * delete memory for all variables
     */
    void deleteMemory()
    {
        for(int i=0; i < numVars; ++i) {
            deleteVarMem (i);
        }
        delete [] values;
    }

    void initialized()
    {
        VisItSetupEnvironment();
        char buffer[1024];
        if (getcwd(buffer, 1024) == NULL) {
            setError(NO_CWD);
        }
        VisItInitializeSocketAndDumpSimFile("libgeodecomp", "",
            buffer, NULL, NULL, NULL);

        checkVisitState();
    }

    void allDone()
    {}


    /**
     * check if there is input from visit
     */
    void checkVisitState()
    {
        // fixme: no do-loops
        // fixme: function too long
        do {
            if (error != 0) {
                break;
            }
            blocking = (runMode == VISIT_SIMMODE_RUNNING) ? 0 : 1;
            visitState = VisItDetectInput(blocking, -1);
            if (visitState <= -1) {
                std::cerr << "Can’t recover from error!" << std::endl;
                error = visitState;
                runMode = VISIT_SIMMODE_RUNNING;
                break;
            } else if (visitState == 0) {
                /* There was no input from VisIt, return control to sim. */
                break;
            } else if (visitState == 1) {
                /* VisIt is trying to connect to sim. */
                if (VisItAttemptToCompleteConnection()) {
                    std::cout << "VisIt connected" << std::endl;

                    initMemory();

                    VisItSetCommandCallback(ControlCommandCallback<CELL_TYPE>,
                            reinterpret_cast<void*>(this));
                    VisItSetGetMetaData(SimGetMetaData, reinterpret_cast<void*>(this));

                    typedef typename MESH_TYPE::template GetMesh<CELL_TYPE, DIMENSIONS> Mesh;
                    VisItSetGetMesh(Mesh::SimGetMesh, reinterpret_cast<void*>(this));

                    VisItSetGetVariable(callSetGetVariable, reinterpret_cast<void*>(this));
                } else {
                    char *visitError = VisItGetLastError();
                    std::cerr << "VisIt did not connect: " << visitError << std::endl;
                }
            } else if (visitState == 2) {
                /* VisIt wants to tell the engine something. */
                runMode = VISIT_SIMMODE_STOPPED;
                if (!VisItProcessEngineCommand()) {
                    /* Disconnect on an error or closed connection. */
                    std::cout << "VisIt disconnected" << std::endl;
                    VisItDisconnect();

                    deleteMemory();

                    /* Start running again if VisIt closes. */
                    runMode = VISIT_SIMMODE_RUNNING;
                    break;
                }
                if (runMode == SIMMODE_STEP) {
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
    static visit_handle callSetGetVariable(
        int domain,
        const char *name,
        void *cbdata)
    {
        typedef SetGetVariable<double, SetDataDouble> VisitDataDouble;
        typedef SetGetVariable<int, SetDataInt> VisitDataInt;
        typedef SetGetVariable<float, SetDataFloat> VisitDataFloat;
        typedef SetGetVariable<char, SetDataChar> VisitDataChar;
        typedef SetGetVariable<long, SetDataLong> VisitDataLong;

        SVW *simData = reinterpret_cast<SVW*>(cbdata);

        for(int i=0; i < simData->getNumVars(); ++i) {
            if (strcmp("DOUBLE", simData->dataAccessors[i]->getType().c_str()) == 0) {
                return VisitDataDouble::SimGetVariable(domain, name, simData);
            } else if (strcmp("INT", simData->dataAccessors[i]->getType().c_str()) == 0) {
                return VisitDataInt::SimGetVariable(domain, name, simData);
            } else if (strcmp("FLOAT", simData->dataAccessors[i]->getType().c_str()) == 0) {
                return VisitDataFloat::SimGetVariable(domain, name, simData);
            } else if (strcmp("CHAR", simData->dataAccessors[i]->getType().c_str()) == 0) {
                return VisitDataChar::SimGetVariable(domain, name, simData);
            } else if (strcmp("LONG", simData->dataAccessors[i]->getType().c_str()) == 0) {
                return VisitDataLong::SimGetVariable(domain, name, simData);
            } else {
                simData->setError(UNKNOWN_TYPE);
            }
        }
        return VISIT_INVALID_HANDLE;
    }

    /**
     *
     */
    template<typename T, typename SETDATAFUNC>
    class SetGetVariable
    {
    public:
        // fixme: too long
        static visit_handle SimGetVariable(
            int domain,
            const char *name,
            SVW *simData)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            CoordBox<DIMENSIONS> box = simData->getGrid()->boundingBox();
            unsigned int size = box.size();

            if(VisIt_VariableData_alloc(&h) == VISIT_OKAY) {
                if(simData->variableMap.find(name) != simData->variableMap.end()) {
                    int num = simData->variableMap[name];
                    T *value  = (T *) simData->values[num];
                    unsigned j = 0;
                    for(typename CoordBox<DIMENSIONS>::Iterator i = box.begin();
                            i != box.end(); ++i) {
                        simData->dataAccessors[num]->getFunction(
                                simData->getGrid()->at(*i), reinterpret_cast<void*>(&value[j]));
                        ++j;
                    }
                    SETDATAFUNC()(h, VISIT_OWNER_SIM, 1, size, value);
                } else {
                    VisIt_VariableData_free(h);
                    h = VISIT_INVALID_HANDLE;
                }
            }
            return h;
        }
    };

    class SetDataDouble 
    {
    public:
        void operator()(visit_handle obj, int owner, int ncomps, int ntuples, double *ptr) {
            VisIt_VariableData_setDataD(obj, owner, ncomps, ntuples, ptr);
        }
    };

    class SetDataInt 
    {
    public:
        void operator()(visit_handle obj, int owner, int ncomps, int ntuples, int  *ptr) {
            VisIt_VariableData_setDataI(obj, owner, ncomps, ntuples, ptr);
        }
    };

    // fixme: unify into one class?
    class SetDataFloat 
    {
    public:
        void operator()(visit_handle obj, int owner, int ncomps, int ntuples, float *ptr) {
            VisIt_VariableData_setDataF(obj, owner, ncomps, ntuples, ptr);
        }
    };

    class SetDataChar 
    {
    public:
        void operator()(visit_handle obj, int owner, int	ncomps, int ntuples, char *ptr) {
            VisIt_VariableData_setDataC(obj, owner, ncomps, ntuples, ptr);
        }
    };

    class SetDataLong 
    {
    public:
        void operator()(visit_handle obj, int owner, int	ncomps, int ntuples, long *ptr) {
            VisIt_VariableData_setDataL(obj, owner, ncomps, ntuples, ptr);
        }
    };

    /**
     * set meta data for visit:
     *      - variable type (only zonal scalar variable at the moment)
     */
    static visit_handle SimGetMetaData(void *cbdata) {
        visit_handle md = VISIT_INVALID_HANDLE;
        SVW *simData = reinterpret_cast<SVW*>(cbdata);

        // fixme: too long
        if (VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY) {
            if (simData->runMode == VISIT_SIMMODE_STOPPED) {
                VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_STOPPED);
            } else if (simData->runMode == SIM_STOPPED) {
                VisIt_SimulationMetaData_setMode(md,  VISIT_SIMMODE_RUNNING);
            } else {
                VisIt_SimulationMetaData_setMode(md,  VISIT_SIMMODE_RUNNING);
            }
            VisIt_SimulationMetaData_setCycleTime(md, simData->getStep(), 0);

            visit_handle m1 = VISIT_INVALID_HANDLE;
            visit_handle m2 = VISIT_INVALID_HANDLE;
            visit_handle vmd = VISIT_INVALID_HANDLE;

            if (DIMENSIONS == 2) {
                /* Set the first mesh's properties.*/
                if (VisIt_MeshMetaData_alloc(&m1) == VISIT_OKAY) {
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
                for(std::map<std::string, int>::iterator it = simData->variableMap.begin();
                    it != simData->variableMap.end(); ++it) {
                    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY) {
                        VisIt_VariableMetaData_setName(vmd, it->first.c_str());
                        VisIt_VariableMetaData_setMeshName(vmd, "mesh2d");
                        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

                        VisIt_SimulationMetaData_addVariable(md, vmd);
                    }
                }
            }

            if (DIMENSIONS == 3) {
                /* Set the second mesh's properties for 3d.*/
                if (VisIt_MeshMetaData_alloc(&m2) == VISIT_OKAY) {
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
                for(std::map<std::string, int>::iterator it = simData->variableMap.begin();
                    it != simData->variableMap.end(); ++it) {
                    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY) {
                        VisIt_VariableMetaData_setName(vmd, it->first.c_str());
                        VisIt_VariableMetaData_setMeshName(vmd, "mesh3d");
                        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

                        VisIt_SimulationMetaData_addVariable(md, vmd);
                    }
                }
            }
        } else {
            return VISIT_INVALID_HANDLE;
        }

        const char *cmd_names[] = { "halt", "step", "run" };
        for (int i = 0; i < sizeof(cmd_names) / sizeof(const char *);
                ++i) {
            visit_handle cmd = VISIT_INVALID_HANDLE;

            if (VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY) {
                VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
                VisIt_SimulationMetaData_addGenericCommand(md, cmd);
            }
        }
        return md;
    }

};

class RectilinearMesh 
{
public:
    static int getMeshType() 
    {
        return VISIT_MESHTYPE_RECTILINEAR;
    }

    template<typename CELL, int DIM>
    class GetMesh;

    template<typename CELL>
    class GetMesh<CELL, 2> 
    {
    public:
        static visit_handle SimGetMesh(
            int domain, 
            const char *name, 
            void *cdata) 
        {
            // fixme: too long
            visit_handle h = VISIT_INVALID_HANDLE;
            VisitWriter<CELL, RectilinearMesh >* simData = reinterpret_cast
                    <VisitWriter<CELL, RectilinearMesh >*>(cdata);

            int dim_x = simData->getGrid()->getDimensions().x() + 1;
            int dim_y = simData->getGrid()->getDimensions().y() + 1;

            if(strcmp(name, "mesh2d") == 0) {
                if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR) {
                    simData->x = reinterpret_cast<double*>(
                        malloc(sizeof(double)*dim_x));
                    simData->y = reinterpret_cast<double*>(
                        malloc(sizeof(double)*dim_y));

                    for (int i = 0; i < dim_x; ++i) {
                        simData->x[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_y; ++i) {
                        simData->y[i] = i*1.0;
                    }

                    visit_handle hxc, hyc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, simData->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, simData->y);
                    VisIt_RectilinearMesh_setCoordsXY(h, hxc, hyc);
                }
            }
            return h;
        }
    };

    template<typename CELL>
    class GetMesh<CELL, 3> 
    {
    public:
        // fixme: unify?
        // fixme: too long
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata) 
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            VisitWriter<CELL, RectilinearMesh >* simData = reinterpret_cast
                    <VisitWriter<CELL, RectilinearMesh >*>(cdata);

            int dim_x = simData->getGrid()->getDimensions().x() + 1;
            int dim_y = simData->getGrid()->getDimensions().y() + 1;
            int dim_z = simData->getGrid()->getDimensions().z() + 1;

            if(strcmp(name, "mesh3d") == 0) {
                if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR) {
                    simData->x = reinterpret_cast<double*>(
                        malloc(sizeof(double)*dim_x));
                    simData->y = reinterpret_cast<double*>(
                        malloc(sizeof(double)*dim_y));
                    simData->z = reinterpret_cast<double*>(
                        malloc(sizeof(double)*dim_z));

                    for (int i = 0; i < dim_x; ++i) {
                        simData->x[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_y; ++i) {
                        simData->y[i] = i*1.0;
                    }
                    for (int i = 0; i < dim_z; ++i) {
                        simData->z[i] = i*1.0;
                    }

                    visit_handle hxc, hyc, hzc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_alloc(&hzc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, simData->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, simData->y);
                    VisIt_VariableData_setDataD(hzc, VISIT_OWNER_VISIT, 1, dim_z, simData->z);
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
class PointMesh
{
public:
    static int getMeshType() {
        return VISIT_MESHTYPE_RECTILINEAR;
    }

    template<typename CELL, int DIM>
    class GetMesh;

    template<typename CELL>
    class GetMesh<CELL, 2> {
    public:
        // fixme: too long
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata) {
            visit_handle h = VISIT_INVALID_HANDLE;
            VisitWriter<CELL, PointMesh >* simData = reinterpret_cast
                    <VisitWriter<CELL, PointMesh >*>(cdata);

            int dim_x = simData->getGrid()->getDimensions().x();
            int dim_y = simData->getGrid()->getDimensions().y();

            unsigned size = simData->getGrid()->boundingBox().size();

            if(strcmp(name, "mesh2d") == 0) {
                if(VisIt_PointMesh_alloc(&h) != VISIT_ERROR) {
                    simData->x = reinterpret_cast<double*>(
                        malloc(sizeof(double)*size));
                    simData->y = reinterpret_cast<double*>(
                        malloc(sizeof(double)*size));

                    unsigned s = 0;
                    for (int j = 0; j < dim_y; ++j) {
                        for (int k = 0; k < dim_x; ++k) {
                            simData->x[s] = k * 1.0;
                            simData->y[s] = j * 1.0;
                            ++s;
                        }
                    }

                    visit_handle hxc, hyc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, size, simData->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, size, simData->y);
                    VisIt_PointMesh_setCoordsXY(h, hxc, hyc);
                }
            }
            return h;
        }
    };

    template<typename CELL>
    class GetMesh<CELL, 3> 
    {
    public:
        // fixme: too long
        static visit_handle SimGetMesh(int domain, const char *name, void *cdata)
        {
            visit_handle h = VISIT_INVALID_HANDLE;
            VisitWriter<CELL, PointMesh >* simData = reinterpret_cast
                    <VisitWriter<CELL, PointMesh >*>(cdata);

            unsigned size = simData->getGrid()->boundingBox().size();

            if(strcmp(name, "mesh3d") == 0) {
                if(VisIt_PointMesh_alloc(&h) != VISIT_ERROR) {
                    simData->x = reinterpret_cast<double*>(
                        malloc(sizeof(double)*size));
                    simData->y = reinterpret_cast<double*>(
                        malloc(sizeof(double)*size));
                    simData->z = reinterpret_cast<double*>(
                        malloc(sizeof(double)*size));

                    CoordBox<3> box = simData->getGrid()->boundingBox();

                    unsigned j = 0;
                    for(typename CoordBox<3>::Iterator i = box.begin();
                            i != box.end(); ++i) {
                        simData->x[j] = i->x() * 1.0;
                        simData->y[j] = i->y() * 1.0;
                        simData->z[j] = i->z() * 1.0;
                        j++;
                    }

                    visit_handle hxc, hyc, hzc;
                    VisIt_VariableData_alloc(&hxc);
                    VisIt_VariableData_alloc(&hyc);
                    VisIt_VariableData_alloc(&hzc);
                    VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, size, simData->x);
                    VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, size, simData->y);
                    VisIt_VariableData_setDataD(hzc, VISIT_OWNER_VISIT, 1, size, simData->z);
                    VisIt_PointMesh_setCoordsXYZ(h, hxc, hyc, hzc);
                }
            }
            return h;
        }
    };
};

// fixme: move to dedicated file?
template < typename CELL_TYPE >
void ControlCommandCallback(
    const char *cmd,
    const char *args, void *cbdata) 
{
    VisitWriter<CELL_TYPE> *simData = (VisitWriter<CELL_TYPE> *) cbdata;

    if (strcmp(cmd, "halt") == 0) {
        simData->setRunMode(VISIT_SIMMODE_STOPPED);
    } else if (strcmp(cmd, "step") == 0) {
        simData->setRunMode(SIMMODE_STEP);
    } else if (strcmp(cmd, "run") == 0) {
        simData->setRunMode(VISIT_SIMMODE_RUNNING);
    }
}

}

#endif
