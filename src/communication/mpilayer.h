#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_MPILAYER_MPILAYER_H
#define LIBGEODECOMP_MPILAYER_MPILAYER_H

#include <mpi.h>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

/**
 * MPILayer is a wrapper that provides a mostly 1:1 identical access
 * to MPI functions, but with a number of convenient twists, where
 * appropriate.
 */
class MPILayer
{
    friend class MPILayerTest;
    friend class ParallelMPILayerTest;
public:
    /**
     * Tags are required to distinguish different transmissions with
     * identical sender, receiver, and communicator. We list all tags
     * in a central place to check for collisions. Mostly having one
     * tag per class should be enough (assuming that those objects
     * communicate with their siblings only). In those cases when we
     * have multiple instances (e.g. two PatchLink connections), an
     * offset to the tag should be used.
     */
    enum Tag {
        // reserve [100, 199], assuming there won't be more than 100
        // links between any two nodes.
        PATCH_LINK = 100,
        PARALLEL_MEMORY_WRITER = 200
    };

    typedef std::map<int, std::vector<MPI_Request> > RequestsMap;

    /**
     * Sets up a new MPILayer. communicator will be used as a scope
     * for all MPI functions, tag will be the default tag passed to
     * all point-to-point communication functions.
     */
    MPILayer(MPI_Comm communicator = MPI_COMM_WORLD, int tag = 0) :
        comm(communicator),
        tag(tag)
    {}

    virtual ~MPILayer()
    {
        waitAll();
    }

    template<typename T>
    inline void send(
        const T *c,
        int dest,
        int num = 1,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        send(c, dest, num, tag, datatype);
    }

    template<typename T>
    inline void send(
        const T *c,
        int dest,
        int num,
        int tag,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI_Request req;
        MPI_Isend(const_cast<T*>(c), num, datatype, dest, tag, comm, &req);
        requests[tag].push_back(req);
    }

    template<typename T>
    inline void recv(
        T *c,
        int src,
        int num = 1,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        recv(c, src, num, tag, datatype);
    }

    template<typename T>
    inline void recv(
        T *c,
        int src,
        int num,
        int tag,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI_Request req;
        MPI_Irecv(c, num, datatype, src, tag, comm, &req);
        requests[tag].push_back(req);
    }

    void cancelAll()
    {
        for (RequestsMap::iterator i = requests.begin();
             i != requests.end();
             ++i) {
            cancel(i->first);
        }
    }

    void cancel(int waitTag)
    {
        std::vector<MPI_Request>& requestVec = requests[waitTag];
        for (std::vector<MPI_Request>::iterator i = requestVec.begin();
             i != requestVec.end(); ++i) {
            MPI_Cancel(&*i);
        }
    }

    /**
     * blocks until all asynchronous communications have been
     * completed.
     */
    void waitAll()
    {
        for (RequestsMap::iterator i = requests.begin();
             i != requests.end();
             ++i) {
            wait(i->first);
        }
    }

    /**
     * waits until those communication requests tagged with waitTag
     * are finished.
     */
    void wait(int waitTag)
    {
        std::vector<MPI_Request>& requestVec = requests[waitTag];
        if(requestVec.size() > 0) {
            MPI_Waitall(requestVec.size(), &requestVec[0], MPI_STATUSES_IGNORE);
        }
        requestVec.clear();
    }

    void testAll()
    {
        for (RequestsMap::iterator i = requests.begin();
             i != requests.end();
             ++i) {
            test(i->first);
        }
    }

    void test(int testTag)
    {
        int flag;
        std::vector<MPI_Request>& requestVec = requests[testTag];
        if(requestVec.size() > 0) {
            MPI_Testall(requestVec.size(), &requestVec[0], &flag, MPI_STATUSES_IGNORE);
        }
    }

    void barrier()
    {
        MPI_Barrier(comm);
    }

    MPI_Comm communicator()
    {
        return comm;
    }

    /**
     * returns the number of nodes in the communicator.
     */
    int size() const
    {
        int ret;
        MPI_Comm_size(comm, &ret);
        return ret;
    }

    /**
     * returns the id number of the current node.
     */
    int rank() const
    {
        int ret;
        MPI_Comm_rank(comm, &ret);
        return ret;
    }

    template<typename T>
    void sendVec(
        const std::vector<T> *vec,
        int dest,
        int waitTag = 0,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI_Request req;
        MPI_Isend(
            &(const_cast<std::vector<T>&>(*vec))[0],
            vec->size(),
            datatype,
            dest,
            tag,
            comm,
            &req);
        requests[waitTag].push_back(req);
    }

    template<typename T>
    void recvVec(
        std::vector<T> *vec,
        int src,
        int waitTag = 0,
        const MPI_Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI_Request req;
        MPI_Irecv(
            &(*vec)[0],
            vec->size(),
            datatype,
            src,
            tag,
            comm,
            &req);
        requests[waitTag].push_back(req);
    }

    /**
     * Sends a region object synchronously to another node.
     */
    template<int DIM>
    void sendRegion(const Region<DIM>& region, int dest)
    {
        unsigned numStreaks = region.numStreaks();
        MPI_Request req;
        MPI_Isend(&numStreaks, 1, MPI_UNSIGNED, dest, tag, comm, &req);
        if(numStreaks > 0) {
            std::vector<Streak<DIM> > buf = region.toVector();
            MPI_Send(&buf[0], numStreaks, Typemaps::lookup<Streak<DIM> >(), dest, tag, comm);
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    /**
     * Receives a region object from another node, also synchronously.
     */
    template<int DIM>
    void recvRegion(Region<DIM> *region, int src)
    {
        unsigned numStreaks;
        MPI_Recv(&numStreaks, 1, MPI_UNSIGNED, src, tag, comm, MPI_STATUS_IGNORE);
        if(numStreaks > 0) {
            std::vector<Streak<DIM> > buf(numStreaks);
            MPI_Recv(&buf[0], numStreaks, Typemaps::lookup<Streak<DIM> >(), src, tag, comm, MPI_STATUS_IGNORE);
            region->clear();
            region->load(buf.begin(), buf.end());
        }
        else {
            region->clear();
        }
    }

    /**
     * Convenience function that will simply return the received
     * Region by value.
     */
    template<int DIM>
    Region<DIM> recvRegion(int src)
    {
        Region<DIM> ret;
        recvRegion(&ret, src);
        return ret;
    }

    template<typename GRID_TYPE, int DIM>
    void recvUnregisteredRegion(GRID_TYPE *stripe,
                                const Region<DIM>& region,
                                int src,
                                int tag,
                                const MPI_Datatype& datatype)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            recv(&((*stripe)[i->origin]), src, i->length(), tag, datatype);
        }
    }

    template<typename GRID_TYPE, int DIM>
    void sendUnregisteredRegion(GRID_TYPE *stripe,
                                const Region<DIM>& region,
                                int dest,
                                int tag,
                                const MPI_Datatype& datatype)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            send(&((*stripe)[i->origin]), dest, i->length(), tag, datatype);
        }
    }

    template<typename T>
    inline std::vector<T> allGather(
        const T& source,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<T> result(size());
        allGather(source, &result, datatype);
        return result;
    }

    template<typename T>
    inline void allGather(
        const T *source,
        T *target,
        int num,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        MPI_Allgather(const_cast<T*>(source), num, datatype, target, num, datatype, comm);
    }

    template<typename T>
    inline void allGather(
        const T& source,
        std::vector<T> *target,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        allGather(&source, &(target->front()), 1, datatype);
    }

    template<typename T>
    inline std::vector<T> allGatherV(
        const T *source,
        const std::vector<int>& lengths,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<T> result(sum(lengths));
        allGatherV(source, lengths, &result, datatype);
        return result;
    }

    template<typename T>
    inline void allGatherV(
        const T *source,
        const std::vector<int>& lengths,
        std::vector<T> *target,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<int> displacements(size());
        displacements[0] = 0;
        for (int i = 0; i < size() - 1; ++i) {
            displacements[i + 1] = displacements[i] + lengths[i];
        }
        MPI_Allgatherv(
            const_cast<T*>(source), lengths[rank()], datatype,
            &(target->front()), const_cast<int*>(&(lengths.front())), &(displacements.front()), datatype,
            comm);
    }

    template<typename T>
    inline std::vector<T> gather(
        const T& item,
        int root,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<T> result(size());
        MPI_Gather(const_cast<T*>(&item), 1, datatype, &(result.front()), 1, datatype, root, comm);
        if (rank() == root) {
            return result;
        } else {
            return std::vector<T>();
        }
    }

    /**
     * Simple wrapper for MPI_Gatherv. lengths is only relevant on
     * root. Expects that target has sufficient capacity.
     */
    template<typename T>
    inline void gatherV(
        const T *source,
        int num,
        const std::vector<int>& lengths,
        int root,
        T *target,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<int> displacements(size());
        if (rank() == root) {
            displacements[0] = 0;
            for (int i = 0; i < size() - 1; ++i) {
                displacements[i + 1] = displacements[i] + lengths[i];
            }
        }

        MPI_Gatherv(
            const_cast<T*>(source),
            num,
            datatype,
            target,
            const_cast<int*>(&lengths[0]),
            const_cast<int*>(&displacements[0]),
            datatype,
            root,
            comm);
    }

    template<typename T>
    inline void gatherV(
        const std::vector<T>& source,
        const std::vector<int>& lengths,
        int root,
        std::vector<T>& target,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        std::vector<int> displacements(size());
        if (rank() == root) {
            displacements[0] = 0;
            for (int i = 0; i < size() - 1; ++i) {
                displacements[i + 1] = displacements[i] + lengths[i];
            }
        }

        T *source_ptr = source.size()>0?const_cast<T*>(&source[0]):0;
        T *target_ptr = target.size()>0?&target[0]:0;

        int *lengths_ptr = lengths.size()>0 ? const_cast<int*>(&lengths[0]) : 0;
        int *displacements_ptr = lengths.size()>0 ? &displacements[0] : 0;

        MPI_Gatherv(
            source_ptr,
            source.size(),
            datatype,
            target_ptr,
            lengths_ptr,
            displacements_ptr,
            datatype,
            root,
            comm);
    }


    /**
     * Broadcasts static size stuff.
     */
    template<typename T>
    inline T broadcast(
        const T& source,
        int root,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        T buff(source);
        MPI_Bcast(&buff, 1, datatype, root, comm);
        return buff;
    }

    template<typename T>
    void broadcast(
        T *buffer,
        int num,
        int root,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        MPI_Bcast(buffer, num, datatype, root, comm);
    }

    template<typename T>
    inline std::vector<T> broadcastVector(
        const std::vector<T>& source,
        int root,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        unsigned size = source.size();
        size = broadcast(size, root);
        std::vector<T> buff(size);

        if (size == 0) {
            return buff;
        }
        if (rank() == root) {
            buff = source;
        }

        MPI_Bcast(&(buff.front()), size, datatype, root, comm);
        return buff;
    }

    template<typename T>
    inline void broadcastVector(
        std::vector<T> *buffer,
        int root,
        const MPI_Datatype& datatype = Typemaps::lookup<T>()) const
    {
        unsigned size = buffer->size();
        size = broadcast(size, root);
        buffer->resize(size);

        if (size > 0) {
            MPI_Bcast(&(buffer->front()), size, datatype, root, comm);
        }
    }

private:
    MPI_Comm comm;
    int tag;
    RequestsMap requests;

    typedef std::pair<const void*, unsigned> ChunkSpec;

    template<class CELL_TYPE, class GRID_TYPE, int DIM>
    class StreakToAddressTranslatingIterator
    {
    public:
        inline StreakToAddressTranslatingIterator(const GRID_TYPE *_grid, typename Region<DIM>::StreakIterator _iter) :
            grid(_grid),
            iter(_iter)
        {}

        inline void operator++()
        {
            ++iter;
        }

        inline const CELL_TYPE *operator*() const
        {
            return &(*grid)[iter->origin];
        }

    private:
        const GRID_TYPE *grid;
        typename Region<DIM>::StreakIterator iter;
    };

    template<int DIM>
    class StreakToLengthTranslatingIterator
    {
    public:
        inline StreakToLengthTranslatingIterator(typename Region<DIM>::StreakIterator _iter) :
            iter(_iter)
        {}

        inline void operator++()
        {
            ++iter;
        }

        inline int operator*() const
        {
            return iter->length();;
        }

    private:
        typename Region<DIM>::StreakIterator iter;
    };

    static bool addressLower(ChunkSpec a, ChunkSpec b)
    {
        return a.first < b.first;
    }

};

}


#endif
#endif
