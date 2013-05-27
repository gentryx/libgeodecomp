#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI

#ifndef LIBGEODECOMP_IO_REMOTESTEERER_STEERERDATA_H
#define LIBGEODECOMP_IO_REMOTESTEERER_STEERERDATA_H

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/dataaccessor.h>
#include <libgeodecomp/misc/supervector.h>
#include <mpi.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

// fixme: use MPILayer
/*
 *  exchange data between nextStep an command functions
 */
template<typename CELL_TYPE>
class SteererData
{
public:
    typedef SuperVector<boost::shared_ptr<DataAccessor<CELL_TYPE> > > DataAccessorVec;

    SteererData(const MPI::Intracomm& comm = MPI::COMM_WORLD) :
        comm(comm)
    {
        finishMutex.lock();
        waitMutex.lock();
    }

    // fixme: why public?
    boost::mutex finishMutex;
    boost::mutex waitMutex;
    std::vector<int> getX;
    std::vector<int> getY;
    std::vector<int> getZ;
    std::vector<int> setX;
    std::vector<int> setY;
    std::vector<int> setZ;
    std::vector<std::string> val;
    std::vector<std::string> var;
    DataAccessorVec dataAccessors;
    MPI::Intracomm comm;

    /**
     * Adds an accessor which allows the VisItWriter to observer another variable.
     */
    void addVariable(DataAccessor<CELL_TYPE> *accessor)
    {
        dataAccessors << boost::shared_ptr<DataAccessor<CELL_TYPE> >(accessor);
    }

    int numVars()
    {
        return dataAccessors.size();
    }

    /**
     * Broadcasts new steering data from root to the other processes.
     */
    void broadcastSteererData()
    {
        if (comm.Get_rank() != 0) {
            getX.clear();
            getY.clear();
            getZ.clear();
            setX.clear();
            setY.clear();
            setZ.clear();
            val.clear();
            var.clear();
        }
        broadcastIntVector(&getX);
        broadcastIntVector(&getY);
        broadcastIntVector(&getZ);
        broadcastIntVector(&setX);
        broadcastIntVector(&setY);
        broadcastIntVector(&setZ);
        broadcastStringVector(&val);
        broadcastStringVector(&var);
    }

  private:
    void broadcastIntVector(std::vector<int> *vec)
    {
        int size = 0;
        int* intBuffer;
        if (comm.Get_rank() == 0) {
            size = vec->size();
            comm.Bcast(&size, 1, MPI::INT, 0);
            comm.Bcast(&(vec->front()), size, MPI::INT, 0);
        }
        else {
            comm.Bcast(&size, 1, MPI::INT, 0);
            intBuffer = new int[size];
            comm.Bcast(intBuffer, size, MPI::INT, 0);
            vec->insert(vec->begin(), intBuffer, intBuffer + size);
            delete[] intBuffer;
        }
    }

    /*
     *
     */
    void broadcastStringVector(std::vector<std::string> *vec)
    {
        int vectorSize = 0;
        int stringSize = 0;
        char* charBuffer;
        if (comm.Get_rank() == 0) {
            vectorSize = vec->size();
            comm.Bcast(&vectorSize, 1, MPI::INT, 0);
            for (int i = 0; i < vectorSize; ++i) {
                stringSize = vec->at(i).size();
                comm.Bcast(&stringSize, 1, MPI::INT, 0);
                char* tmp = const_cast<char*>(vec->at(i).c_str());
                comm.Bcast(tmp, stringSize, MPI::CHAR, 0);
            }
        }
        else {
            comm.Bcast(&vectorSize, 1, MPI::INT, 0);
            for (int i = 0; i < vectorSize; ++i) {
                comm.Bcast(&stringSize, 1, MPI::INT, 0);
                charBuffer = new char[stringSize];
                comm.Bcast(charBuffer, stringSize, MPI::CHAR, 0);
                vec->push_back(charBuffer);
                delete[] charBuffer;
            }
        }
    }
};

}

}

#endif
#endif
