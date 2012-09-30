#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_mpilayer_mpilayer_h_
#define _libgeodecomp_mpilayer_mpilayer_h_

#include <mpi.h>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/mpilayer/typemaps.h>

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

    typedef std::map<int, std::vector<MPI::Request> > RequestsMap;

    MPILayer(MPI::Comm *c = &MPI::COMM_WORLD, int _tag = 0) :
        comm(c),
        tag(_tag)
    {}

    virtual ~MPILayer()
    {
        waitAll();
    }

    template<typename T>
    inline void send(
        const T *c, 
        const int& dest, 
        const int& num = 1,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        send(c, dest, num, tag, datatype);
    }

    template<typename T>
    inline void send(
        const T *c, 
        const int& dest, 
        const int& num,
        const int& tag,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI::Request req = comm->Isend(c, num, datatype, dest, tag);
        requests[tag].push_back(req);
    }
    
    template<typename T>
    inline void recv(
        T *c, 
        const int& src, 
        const int& num = 1,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        recv(c, src, num, tag, datatype);
    }
    
    template<typename T>
    inline void recv(
        T *c, 
        const int& src, 
        const int& num,
        const int& tag,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI::Request req = comm->Irecv(c, num, datatype, src, tag);
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

    void cancel(const int& waitTag) 
    { 
        std::vector<MPI::Request>& requestVec = requests[waitTag];
        for (std::vector<MPI::Request>::iterator i = requestVec.begin();
             i != requestVec.end(); ++i) {
            i->MPI::Request::Cancel();
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
     * waits until those communication requests tagged with @a waitTag
     * are finished.
     */
    void wait(const int& waitTag) 
    { 
        std::vector<MPI::Request>& requestVec = requests[waitTag];
        MPI::Request::Waitall(requestVec.size(), &requestVec[0]);
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

    void test(const int& testTag) 
    { 
        std::vector<MPI::Request>& requestVec = requests[testTag];
        MPI::Request::Testall(requestVec.size(), &requestVec[0]);
    }

    void barrier()
    {
        comm->Barrier();
    }

    MPI::Comm *getCommunicator()
    {
        return comm;
    }
    
    /** 
     * @return the number of nodes in the communicator. 
     */ 
    unsigned size() const
    {
        return comm->Get_size(); 
    }

    /** 
     * @return the id number of the current node. 
     */ 
    unsigned rank() const
    {
        return comm->Get_rank(); 
    } 
    
    template<typename T>
    void sendVec(
        const SuperVector<T> *vec, 
        const int& dest, 
        const int& waitTag = 0, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI::Request req = comm->Isend(
            &(const_cast<SuperVector<T>&>(*vec))[0], 
            vec->size(),
            datatype,
            dest,
            tag);
        requests[waitTag].push_back(req);    
    }

    template<typename T>
    void recvVec(
        SuperVector<T> *vec, 
        const int& src, 
        const int& waitTag = 0,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI::Request req = comm->Irecv(
            &(*vec)[0], 
            vec->size(),
            datatype,
            src,
            tag);
        requests[waitTag].push_back(req);    
    }

    /**
     * Sends a region object synchronously to another node.
     */
    template<int DIM>
    void sendRegion(const Region<DIM>& region, const int& dest)
    {
        unsigned numStreaks = region.numStreaks();
        MPI::Request req = comm->Isend(&numStreaks, 1, MPI::UNSIGNED, dest, tag);
        SuperVector<Streak<DIM> > buf = region.toVector();
        comm->Send(&buf[0], numStreaks, Typemaps::lookup<Streak<DIM> >(), dest, tag);
        req.Wait();
    }

    /**
     * Receives a region object from another node, also synchronously.
     */
    template<int DIM>
    void recvRegion(Region<DIM> *region, const int& src)
    {
        unsigned numStreaks;
        comm->Recv(&numStreaks, 1, MPI::UNSIGNED, src, tag);
        SuperVector<Streak<DIM> > buf(numStreaks);
        comm->Recv(&buf[0], numStreaks, Typemaps::lookup<Streak<DIM> >(), src, tag);
        region->clear();
        region->load(buf.begin(), buf.end());
    }
    
    /**
     * Convenience function that will simply return the received
     * Region by value.
     */
    template<int DIM>
    Region<DIM> recvRegion(const int& src)
    {
        Region<DIM> ret;
        recvRegion(&ret, src);
        return ret;
    }

    template<typename GRID_TYPE, int DIM>
    void recvUnregisteredRegion(GRID_TYPE *stripe, 
                                const Region<DIM>& region, 
                                const int& src, 
                                const int& tag,
                                const MPI::Datatype& datatype)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            recv(&(*stripe).at(i->origin), src, i->length(), tag, datatype);
        }
    }

    template<typename GRID_TYPE, int DIM>
    void sendUnregisteredRegion(GRID_TYPE *stripe, 
                                const Region<DIM>& region, 
                                const int& dest, 
                                const int& tag,
                                const MPI::Datatype& datatype)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            send(&(*stripe).at(i->origin), dest, i->length(), tag, datatype);
        }
    }
        
    template<typename T>
    inline SuperVector<T> allGather(
        const T& source, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        SuperVector<T> result(size());
        allGather(source, &result, datatype);
        return result;
    }

    template<typename T>
    inline void allGather(
        const T *source,
        T *target,
        const int& num,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        comm->Allgather(source, num, datatype, target, num, datatype);
    }
        
    template<typename T>
    inline void allGather(
        const T& source, 
        SuperVector<T> *target, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        allGather(&source, &(target->front()), 1, datatype);
    }

    template<typename T>
    inline SuperVector<T> allGatherV(
        const T *source, 
        const SuperVector<int>& lengths,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        SuperVector<T> result(lengths.sum());
        allGatherV(source, lengths, &result, datatype);
        return result;
    }

    template<typename T>
    inline void allGatherV(
        const T *source, 
        const SuperVector<int>& lengths,
        SuperVector<T> *target, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        SuperVector<int> displacements(size());
        displacements[0] = 0;
        for (int i = 0; i < size() - 1; ++i)
            displacements[i + 1] = displacements[i] + lengths[i];
        comm->Allgatherv(source, lengths[rank()], datatype, &(target->front()), &(lengths.front()), &(displacements.front()), datatype);
    }

    template<typename T>
    inline SuperVector<T> gather(
        const T& item, 
        const unsigned& root, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        SuperVector<T> result(size());
        comm->Gather(&item, 1, datatype, &(result.front()), 1, datatype, root);
        if (rank() == root) {
            return result;
        } else {
            return SuperVector<T>();
        }
    }

    /**
     * scatter static size stuff. (T needs to bring a default constructor)
     */
    template<typename T>
    inline T broadcast(
        const T& source, 
        const unsigned& root, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        T buff(source);
        comm->Bcast(&buff, 1, datatype, root);
        return buff;
    }

    template<typename T>
    inline SuperVector<T> broadcastVector(
        const SuperVector<T>& source, 
        const unsigned& root,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        unsigned size = source.size();
        size = broadcast(size, root);
        SuperVector<T> buff(size);
        if (size == 0) return buff;
        if (rank() == root) buff = source;
        comm->Bcast(&(buff.front()), size, datatype, root);
        return buff;
    }

    template<typename T>
    inline void broadcastVector(
        SuperVector<T> *buff, 
        const unsigned& root,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        unsigned size = buff->size();
        size = broadcast(size, root);
        if (size > 0) 
            comm->Bcast(&(buff->front()), size, datatype, root);
    }

private:
    MPI::Comm *comm;
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
