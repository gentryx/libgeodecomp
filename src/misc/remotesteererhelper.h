#ifndef _libgeodecomp_misc_remotesteererhelper_h_
#define _libgeodecomp_misc_remotesteererhelper_h_

#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/misc/commandserver.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelper {

/*
 *  exchange data between nextStep an command functions
 */
template<typename CELL_TYPE>
struct SteererData {
    SteererData(DataAccessor<CELL_TYPE>** _dataAccessors, int _numVars) :
            dataAccessors(_dataAccessors),
            numVars(_numVars)
    {
        getValue_mutex.lock();
        setValue_mutex.lock();
        finish_mutex.lock();
        wait_mutex.lock();
    }

    boost::mutex getValue_mutex;
    boost::mutex setValue_mutex;
    boost::mutex finish_mutex;
    boost::mutex wait_mutex;
    std::vector<int> get_x;
    std::vector<int> get_y;
    std::vector<int> get_z;
    std::vector<int> set_x;
    std::vector<int> set_y;
    std::vector<int> set_z;
    std::vector<std::string> val;
    std::vector<std::string> var;
    DataAccessor<CELL_TYPE>** dataAccessors;
    int numVars;
};

/*
 *
 */
static double mystrtod(const char *ptr)
{
    errno = 0;
    double val = 0;
    char *endptr;
    val = strtod(ptr, &endptr);
    if ((errno != 0) || (ptr == endptr))
    {
        throw(std::invalid_argument("bad paramter"));
    }
    return val;
}

/*
 *
 */
static float mystrtof(const char *ptr)
{
    errno = 0;
    float val = 0;
    char *endptr;
    val = strtof(ptr, &endptr);
    if ((errno != 0) || (ptr == endptr))
    {
        throw(std::invalid_argument("bad paramter"));
    }
    return val;
}

/*
 *
 */
static long mystrtol(const char *ptr)
{
    errno = 0;
    long val = 0;
    char *endptr;
    val = strtol(ptr, &endptr, 10);
    if ((errno != 0) || (ptr == endptr))
    {
        throw(std::invalid_argument("bad paramter"));
    }
    return val;
}

/*
 *
 */
static int mystrtoi(const char *ptr)
{
    errno = 0;
    long val = 0;
    char *endptr;
    val = strtol(ptr, &endptr, 10);
    if ((errno != 0) || (val < INT_MIN) || (val > INT_MAX) || (ptr == endptr))
    {
        throw(std::invalid_argument("bad paramter"));
    }
    return static_cast<int>(val);
}


/*
 *
 */
template<typename CELL_TYPE, int DIM>
static void executeGetRequests(typename Steerer<CELL_TYPE>::GridType*,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>&,
            const unsigned&,
            CommandServer::Session*,
            void *);

/*
 *
 */
template<typename CELL_TYPE, int DIM>
static void executeSetRequests(typename Steerer<CELL_TYPE>::GridType*,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>&,
            const unsigned&,
            CommandServer::Session*,
            void *);

/*
 *
 */
template<typename CELL_TYPE, int DIM>
class SteererControl
{
public:
    virtual void operator()(typename Steerer<CELL_TYPE>::GridType*,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>&,
            const unsigned&, CommandServer::Session*, void*) = 0;
};

/*
 *
 */
template<typename CELL_TYPE, int DIM>
class ExtendedSteererControlStub : SteererControl<CELL_TYPE, DIM>{
public:
    void operator()(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const unsigned& step,
            CommandServer::Session* session,
            void *data)
    {
        // do nothing, it's a stub
    }
};

template<typename CELL_TYPE, int DIM,
         typename EXTENDED = ExtendedSteererControlStub<CELL_TYPE, DIM> >
class DefaultSteererControl : SteererControl<CELL_TYPE, DIM>{
public:
    void operator()(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const unsigned& step,
            CommandServer::Session* session,
            void *data)
    {
        SteererData<CELL_TYPE>* sdata = (SteererData<CELL_TYPE>*) data;

        if (sdata->finish_mutex.try_lock())
        {
            EXTENDED()(grid, validRegion, step, session, data);

            if (sdata->setValue_mutex.try_lock())
            {
                executeSetRequests<CELL_TYPE, DIM>(
                        grid, validRegion, step, session, data);
            }

            if (sdata->getValue_mutex.try_lock())
            {
                executeGetRequests<CELL_TYPE, DIM>(
                        grid, validRegion, step, session, data);
            }

            session->sendMessage("step finished, all jobs done\n");
            sdata->wait_mutex.unlock();
        }
    }
};


/*
 *
 */
template<typename T, typename CELL_TYPE, int DIM>
class Request;

/*
 *
 */
template<typename T, typename CELL_TYPE>
class Request<T, CELL_TYPE,2>
{
public:
    /*
     *
     */
    static bool validCoords(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const int x, const int y, const int z)
    {
        return validRegion.boundingBox().inBounds(Coord<2>(x, y));
    }

    /*
     *
     */
    static int associateAccessor (DataAccessor<CELL_TYPE>** dataAccessors,
                           int num, std::string name)
    {
        for (int i = 0; i < num; ++i)
        {
            if ((dataAccessors[i]->getName()).compare(name) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    /*
     *
     */
    static T valueRequest(DataAccessor<CELL_TYPE>* dataAccessor,
            typename Steerer<CELL_TYPE>::GridType *grid,
            const int x, const int y, const int z)
    {
        T value;
        value = 0;
        dataAccessor->getFunction(grid->at(Coord<2>(x, y)),
                                  reinterpret_cast<void*>(&value));
        return value;
    }

    /*
     *
     */
    static void mutationRequest(DataAccessor<CELL_TYPE>* dataAccessor,
            typename Steerer<CELL_TYPE>::GridType *grid,
            const int x, const int y, const int z, T value)
    {
        dataAccessor->setFunction(&(grid->at(Coord<2>(x, y))),
                                  reinterpret_cast<void*>(&value));
    }
};

/*
 *
 */
template<typename T, typename CELL_TYPE>
class Request<T, CELL_TYPE, 3>
{
public:
    /*
     *
     */
    static bool validCoords(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const int x, const int y, const int z)
    {
        return validRegion.boundingBox().inBounds(Coord<3>(x, y, z));
    }

    /*
     *
     */
    static int associateAccessor (DataAccessor<CELL_TYPE>** dataAccessors,
                           int num, std::string name)
    {
        for (int i = 0; i < num; ++i)
        {
            if ((dataAccessors[i]->getName()).compare(name) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    /*
     *
     */
    static T valueRequest(DataAccessor<CELL_TYPE>* dataAccessor,
            typename Steerer<CELL_TYPE>::GridType *grid,
            const int x, const int y, const int z)
    {
        T value;
        value = 0;
        dataAccessor->getFunction(grid->at(Coord<3>(x, y, z)),
                                  reinterpret_cast<void*>(&value));
        return value;
    }

    /*
     *
     */
    static void mutationRequest(DataAccessor<CELL_TYPE>* dataAccessor,
            typename Steerer<CELL_TYPE>::GridType *grid,
            const int x, const int y, const int z, T value)
    {
        dataAccessor->setFunction(&(grid->at(Coord<3>(x, y, z))),
                                  reinterpret_cast<void*>(&value));
    }
};

/*
 *
 */
template<typename CELL_TYPE, int DIM>
static void executeGetRequests(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const unsigned& step,
            CommandServer::Session* session,
            void *data)
{
    SteererData<CELL_TYPE>* sdata = (SteererData<CELL_TYPE>*) data;

    for (int j = 0; j < sdata->get_x.size(); ++j)
    {
        for (int i = 0; i < sdata->numVars; ++i)
        {
            if ((DIM == 3) && (sdata->get_z[j] < 0))
            {
                session->sendMessage("3 dimensional coords needed\n");
                break;
            }
            if (!Request<int, CELL_TYPE, DIM>::validCoords(
                    grid, validRegion, sdata->get_x[j],
                    sdata->get_y[j], sdata->get_z[j]))
            {
                std::string msg = "no valid coords: ";
                msg += boost::to_string(sdata->get_x[j]) + " ";
                msg += boost::to_string(sdata->get_y[j]);
                if (DIM == 3)
                {
                    msg += " " + boost::to_string(sdata->get_z[j]);
                }
                msg += "\n";
                session->sendMessage(msg);
                break;
            }
            std::string output = sdata->dataAccessors[i]->getName();
            output += "[" + boost::to_string(sdata->get_x[j]) + "][";
            output += boost::to_string(sdata->get_y[j]) + "]";
            if (grid->DIM == 3)
            {
                output += "[";
                output += boost::to_string(sdata->get_z[j]) + "]";
            }
            output += " = ";

            if (strcmp("DOUBLE", sdata->dataAccessors[i]->
                                 getType().c_str()) == 0)
            {
                double value = Request<double, CELL_TYPE, DIM>::
                        valueRequest(
                            sdata->dataAccessors[i], grid,
                            sdata->get_x[j], sdata->get_y[j],
                            sdata->get_z[j]);
                output += boost::to_string(value);
            }
            else if (strcmp("INT", sdata->dataAccessors[i]->
                                   getType().c_str()) == 0)
            {
                int value = Request<int, CELL_TYPE, DIM>::
                        valueRequest(
                            sdata->dataAccessors[i], grid,
                            sdata->get_x[j], sdata->get_y[j],
                            sdata->get_z[j]);
                output += boost::to_string(value);
            }
            else if (strcmp("FLOAT", sdata->dataAccessors[i]->
                                     getType().c_str()) == 0)
            {
                float value = Request<float, CELL_TYPE, DIM>::
                        valueRequest(
                            sdata->dataAccessors[i], grid,
                            sdata->get_x[j], sdata->get_y[j],
                            sdata->get_z[j]);
                output += boost::to_string(value);
            }
            else if (strcmp("CHAR", sdata->dataAccessors[i]->
                                    getType().c_str()) == 0)
            {
                char value = Request<char, CELL_TYPE, DIM>::
                        valueRequest(
                            sdata->dataAccessors[i], grid,
                            sdata->get_x[j], sdata->get_y[j],
                            sdata->get_z[j]);
                output += boost::to_string(value);
            }
            else if (strcmp("LONG", sdata->dataAccessors[i]->
                                    getType().c_str()) == 0)
            {
                long value = Request<long, CELL_TYPE, DIM>::
                        valueRequest(
                            sdata->dataAccessors[i], grid,
                            sdata->get_x[j], sdata->get_y[j],
                            sdata->get_z[j]);
                output += boost::to_string(value);
            }
            output += "\n";
            session->sendMessage(output);
        }
    }
    sdata->get_x.clear();
    sdata->get_y.clear();
    sdata->get_z.clear();
}

/*
 *
 */
template<typename CELL_TYPE, int DIM>
static void executeSetRequests(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const unsigned& step,
            CommandServer::Session* session,
            void *data)
{
    SteererData<CELL_TYPE>* sdata = (SteererData<CELL_TYPE>*) data;

    for (int j = 0; j < sdata->set_x.size(); ++j)
    {
        int accessor;

        if ((DIM == 3) && (sdata->set_z[j] < 0))
        {
            session->sendMessage("3 dimensional coords needed\n");
            continue;
        }
        if (!Request<int, CELL_TYPE, DIM>::validCoords(
                grid, validRegion, sdata->set_x[j],
                sdata->set_y[j], sdata->set_z[j]))
        {
            std::string msg = "no valid coords: ";
            msg += boost::to_string(sdata->set_x[j]) + " ";
            msg += boost::to_string(sdata->set_y[j]);
            if (DIM == 3)
            {
                msg += " " + boost::to_string(sdata->set_z[j]);
            }
            msg += "\n";
            session->sendMessage(msg);
            continue;
        }
        accessor = Request<int, CELL_TYPE, DIM>::associateAccessor(
               sdata->dataAccessors, sdata->numVars, sdata->var[j]);
        if (accessor < 0)
        {
            std::string msg = "no valid variable: ";
            msg += sdata->var[j] + "\n";
            session->sendMessage(msg);
            continue;
        }
        std::string output = sdata->dataAccessors[accessor]->getName();
        output += " set [" + boost::to_string(sdata->set_x[j]) + "][";
        output += boost::to_string(sdata->set_y[j]) + "]";
        if (grid->DIM == 3)
        {
            output += "[";
            output += boost::to_string(sdata->set_z[j]) + "]";
        }
        output += " to ";

        if (strcmp("DOUBLE", sdata->dataAccessors[accessor]->
                             getType().c_str()) == 0)
        {
            double value;
            try
            {
                value = mystrtod(sdata->val[j].c_str());
            }
            catch (std::exception& e)
            {
                std::string msg = "bad value: " + sdata->val[j] + "\n";
                session->sendMessage(msg);
                continue;
            }
            Request<double, CELL_TYPE, DIM>::
                    mutationRequest(
                        sdata->dataAccessors[accessor], grid,
                        sdata->set_x[j], sdata->set_y[j],
                        sdata->set_z[j], value);
            session->sendMessage(output + sdata->val[j] + "\n");
        }
        else if (strcmp("INT", sdata->dataAccessors[accessor]->
                               getType().c_str()) == 0)
        {
            int value;
            try
            {
                value = mystrtoi(sdata->val[j].c_str());
            }
            catch (std::exception& e)
            {
                std::string msg = "bad value: " + sdata->val[j] + "\n";
                session->sendMessage(msg);
                continue;
            }
            Request<int, CELL_TYPE, DIM>::
                    mutationRequest(
                        sdata->dataAccessors[accessor], grid,
                        sdata->set_x[j], sdata->set_y[j],
                        sdata->set_z[j], value);
            session->sendMessage(output + sdata->val[j] + "\n");
        }
        else if (strcmp("FLOAT", sdata->dataAccessors[accessor]->
                                 getType().c_str()) == 0)
        {
            float value;
            try
            {
                value = mystrtof(sdata->val[j].c_str());
            }
            catch (std::exception& e)
            {
                std::string msg = "bad value: " + sdata->val[j] + "\n";
                session->sendMessage(msg);
                continue;
            }
            Request<float, CELL_TYPE, DIM>::
                    mutationRequest(
                        sdata->dataAccessors[accessor], grid,
                        sdata->set_x[j], sdata->set_y[j],
                        sdata->set_z[j], value);
            session->sendMessage(output + sdata->val[j] + "\n");
        }
        else if (strcmp("CHAR", sdata->dataAccessors[accessor]->
                                getType().c_str()) == 0)
        {
            char value;
            value = sdata->val[j][0];
            Request<char, CELL_TYPE, DIM>::
                    mutationRequest(
                        sdata->dataAccessors[accessor], grid,
                        sdata->set_x[j], sdata->set_y[j],
                        sdata->set_z[j], value);
            session->sendMessage(output + sdata->val[j][0] + "\n");
        }
        else if (strcmp("LONG", sdata->dataAccessors[accessor]->
                                getType().c_str()) == 0)
        {
            long value;
            try
            {
                value = mystrtol(sdata->val[j].c_str());
            }
            catch (std::exception& e)
            {
                std::string msg = "bad value: " + sdata->val[j] + "\n";
                session->sendMessage(msg);
                continue;
            }
            Request<long, CELL_TYPE, DIM>::
                    mutationRequest(
                        sdata->dataAccessors[accessor], grid,
                        sdata->set_x[j], sdata->set_y[j],
                        sdata->set_z[j], value);
            session->sendMessage(output + sdata->val[j] + "\n");
        }
    }
    sdata->set_x.clear();
    sdata->set_y.clear();
    sdata->set_z.clear();
    sdata->val.clear();
    sdata->var.clear();
}

}

}

#endif
