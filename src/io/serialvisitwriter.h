/**
 * TODO:    check parameter references and make them clearer
 * TODO:    code style!
 * TODO:    exit/error treatment
 * TODO:    support other mesh types(point meshes)
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

#include <stdio.h>

#define debugout(string) std::cout << string << std::endl;

namespace LibGeoDecomp
{

template < typename CELL_TYPE, int NUM >
void ControlCommandCallback(const char *cmd,
                            const char *args, void *cbdata);

    /**
     * contains information about cell variables
     *
     * generated in simulation
     */
    template<typename CELL, int INDEX>
    class DataSelector;

#define DEFINE_DATASELECTOR(CELL, INDEX, MEMBER_TYPE, MEMBER_NAME)          \
    namespace LibGeoDecomp{                                                 \
        template<>                                                          \
        class DataSelector<CELL, INDEX>                       \
        {                                                                   \
        public:                                                             \
            typedef MEMBER_TYPE type;                                       \
            MEMBER_TYPE getValue(CELL object)                               \
            {                                                               \
                return object.MEMBER_NAME;                                  \
            }                                                               \
                                                                            \
            std::string getName()                                           \
            {                                                               \
                return #MEMBER_NAME;                                        \
            }                                                               \
                                                                            \
            static int getSize()                                            \
            {                                                               \
                return sizeof(MEMBER_TYPE);                                 \
            }                                                               \
        };                                                                  \
    }                                                                       \

template < typename CELL_TYPE, int NUM >
class SerialVisitWriter:public Writer < CELL_TYPE >
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef SerialVisitWriter<CELL_TYPE, NUM> SVW;
    static const int DIMENSIONS = Topology::DIMENSIONS;


    using Writer<CELL_TYPE>::sim;
    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    /**
     * constructor
     */
    SerialVisitWriter(const std::string & _prefix,
               MonolithicSimulator < CELL_TYPE > *_sim,
               int _numVars,
               const unsigned &_period = 1)
               : Writer < CELL_TYPE > (_prefix, _sim, _period),
               numVars(_numVars)
    {
        runMode = 0;
        blocking = 0;
        visitstate = 0;
        error = 0;
    }

    /**
     *
     */
    virtual void initialized()
    {
        values = new void*[numVars];

        unsigned size = sim->getGrid()->boundingBox().size();
        SelectorAction<CELL_TYPE, NUM>()(size, values);

        for(int i=0; i < numVars; ++i)
        if (error != 0)
        {
            // TODO: an neue template Methode anpassen
            //variableMap[DataSelector<CELL_TYPE, i>::getName()] = i;
            //DataSelector<CELL_TYPE, i>::initMemory(1);
            if (error != 0)
            {
                return;
            }
            return;
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
    virtual void stepFinished()
    {
        std::cout << "finished step: " << sim->getStep() << std::endl;

        VisItTimeStepChanged();

        if ((sim->getStep() % period) == 0)
        {
            VisItUpdatePlots();
        }

        if (error != 0)
        {
            return;
        }

        checkVisitState();
    }

    /**
     *
     */
    virtual void allDone()
    {
        //TODO: fix free
        //free(this->values);
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

private:
    int blocking;
    int visitstate;
    int error;
    int runMode;
    int dataNumber;
    int numVars;
    void **values;
    std::map<std::string, int> variableMap;

    /**
     * get memory where visualization data are stored
     */
    template<typename CELL, int INDEX>
    class SelectorAction {
    public:
        void operator()(unsigned size, void* address[])
        {
            SelectorAction<CELL, INDEX - 1>()(size, address);
            typedef typename DataSelector<CELL_TYPE, INDEX>::type dataType;
            address[INDEX] = new dataType[size];
        }
    };

    /**
     *
     */
    template<typename CELL>
    class SelectorAction<CELL, -1> {
    public:
        void operator()(unsigned size, void* address[])
        {
        }
    };

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
                            (ControlCommandCallback<CELL_TYPE, NUM>,
                                    (void *) this);
                    /* Register data access callbacks */
                    VisItSetGetMetaData(SimGetMetaData, reinterpret_cast<void*>(this));
                    if (DIMENSIONS == 2)
                    {
                        VisItSetGetMesh(SimGetMesh2D, (void *)&this->sim->getGrid()->getDimensions());
                    }
                    else
                    {
                        VisItSetGetMesh(SimGetMesh3D, (void *)&this->sim->getGrid()->getDimensions());
                    }

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
        SVW *sim_data = reinterpret_cast<SVW*>(cbdata);

        for(int i=0; i < sim_data->getNumVars(); ++i)
        {
            // TODO: an neue template Methode anpassen
            //typedef DataSelector<CELL_TYPE, i>::type dataType;
            //return VisitData2<dataType, SetDataWrapper<NULL, dataType> >
            //                ::SimGetVariable(domain, name, sim_data);
            return VISIT_INVALID_HANDLE;
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
            const GridType *grid = sim_data->sim->getGrid();
            CoordBox<DIMENSIONS> box = grid->boundingBox();
            unsigned int size = box.size();

            if(VisIt_VariableData_alloc(&h) == VISIT_OKAY)
            {
                if(sim_data->variableMap.find(name) != sim_data->variableMap.end())
                {
                    int num = sim_data->variableMap[name];
                    T *value  = (T *) sim_data->values[num];
                    int j = 0;
                    for(typename CoordBox<DIMENSIONS>::Iterator i = box.begin();
                            i != box.end(); ++i)
                    {
                        // TODO: an neue template Methode anpassen
                        //value[j] = DataSelector<CELL_TYPE, num>::getValue(grid->at(*i));
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
    template<typename FOO, class T>
    class SetDataWrapper;

    /**
     *
     */
    template<typename FOO>
    class SetDataWrapper<FOO, double> : private SetDataDouble
    {
    };

    /**
     *
     */
    template<typename FOO>
    class SetDataWrapper<FOO, int> : private SetDataInt
    {
    };

    /**
     *
     */
    template<typename FOO>
    class SetDataWrapper<FOO, float> : private SetDataFloat
    {
    };

    /**
     *
     */
    template<typename FOO>
    class SetDataWrapper<FOO, char> : private SetDataChar
    {
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
            VisIt_SimulationMetaData_setCycleTime(md, sim_data->sim->getStep(), 0);

            visit_handle m1 = VISIT_INVALID_HANDLE;
            visit_handle vmd = VISIT_INVALID_HANDLE;

            /* Set the first mesh's properties.*/
            if (VisIt_MeshMetaData_alloc(&m1) == VISIT_OKAY)
            {
                /* Set the mesh's properties.*/
                VisIt_MeshMetaData_setName(m1, "mesh");
                VisIt_MeshMetaData_setMeshType(m1, VISIT_MESHTYPE_RECTILINEAR);
                VisIt_MeshMetaData_setTopologicalDimension(m1, 3);
                VisIt_MeshMetaData_setSpatialDimension(m1, 3);
                VisIt_MeshMetaData_setXUnits(m1, "cm");
                VisIt_MeshMetaData_setYUnits(m1, "cm");
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
                    VisIt_VariableMetaData_setMeshName(vmd, "mesh");
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                    VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

                    VisIt_SimulationMetaData_addVariable(md, vmd);
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

    /**
     *
     */
    static visit_handle SimGetMesh2D(int domain, const char *name, void *cdata)
    {
        visit_handle h = VISIT_INVALID_HANDLE;
        Coord<2> *coords = reinterpret_cast<Coord<2> *>(cdata);

        int dim_x = coords->x() + 1;
        int dim_y = coords->y() + 1;

        if(strcmp(name, "mesh") == 0)
        {
            if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
            {
                double *x;
                double *y;

                // TODO: free?
                x = new double[dim_x];
                y = new double[dim_y];

                for (int i = 0; i < dim_x; ++i)
                {
                    x[i] = i*1.0;
                }
                for (int i = 0; i < dim_y; ++i)
                {
                    y[i] = i*1.0;
                }

                visit_handle hxc, hyc;
                VisIt_VariableData_alloc(&hxc);
                VisIt_VariableData_alloc(&hyc);
                VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, x);
                VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, y);
                VisIt_RectilinearMesh_setCoordsXY(h, hxc, hyc);

                //VisIt_RectilinearMesh_setRealIndices(h, minRealIndex, maxRealIndex);
            }
        }
        return h;
    }

    /**
     *
     */
    static visit_handle SimGetMesh3D(int domain, const char *name, void *cdata)
    {
        visit_handle h = VISIT_INVALID_HANDLE;
        Coord<3> *coords = reinterpret_cast<Coord<3> *>(cdata);

        int dim_x = coords->x() + 1;
        int dim_y = coords->y() + 1;
        int dim_z = coords->z() + 1;

        if(strcmp(name, "mesh") == 0)
        {
            if(VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
            {
                double *x;
                double *y;
                double *z;

                x = new double[dim_x];
                y = new double[dim_y];
                z = new double[dim_z];

                for (int i = 0; i < dim_x; ++i)
                {
                    x[i] = i*1.0;
                }
                for (int i = 0; i < dim_y; ++i)
                {
                    y[i] = i*1.0;
                }
                for (int i = 0; i < dim_z; ++i)
                {
                    z[i] = i*1.0;
                }

                visit_handle hxc, hyc, hzc;
                VisIt_VariableData_alloc(&hxc);
                VisIt_VariableData_alloc(&hyc);
                VisIt_VariableData_alloc(&hzc);
                VisIt_VariableData_setDataD(hxc, VISIT_OWNER_VISIT, 1, dim_x, x);
                VisIt_VariableData_setDataD(hyc, VISIT_OWNER_VISIT, 1, dim_y, y);
                VisIt_VariableData_setDataD(hzc, VISIT_OWNER_VISIT, 1, dim_z, z);
                VisIt_RectilinearMesh_setCoordsXYZ(h, hxc, hyc, hzc);

                //VisIt_RectilinearMesh_setRealIndices(h, minRealIndex, maxRealIndex);
            }
        }
        return h;
    }
};

/**
 *
 */
template < typename CELL_TYPE, int NUM >
void ControlCommandCallback(const char *cmd,
                            const char *args, void *cbdata)
{
    SerialVisitWriter<CELL_TYPE, NUM> *sim_data = (SerialVisitWriter<CELL_TYPE, NUM> *) cbdata;

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
