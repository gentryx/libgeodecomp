#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_mpilayer_mpilayer_h_
#define _libgeodecomp_mpilayer_mpilayer_h_

#include <mpi.h>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/mpilayer/typemaps.h>

namespace LibGeoDecomp {

class MPILayer
{
    friend class MPILayerTest;
    friend class ParallelMPILayerTest;
public:

    class MPIRegion {
        friend class MPILayer;        
    public:
        ~MPIRegion() {
            indices.Free();
        }

    private:
        MPI::Datatype indices;
    };

    typedef std::map<int, std::vector<MPI::Request> > RequestsMap;
    typedef boost::shared_ptr<MPIRegion> MPIRegionPointer;

    MPILayer(MPI::Comm *c = &MPI::COMM_WORLD) :
        comm(c),
        tag(0)
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
    
    /**
     * sends a sub-box of @a grid to @a dest asynchronously.
     */
    template<typename T>
    void sendGridBox(
        const Grid<T> *grid, 
        const CoordBox<2>& box, 
        const int& dest,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        for (CoordBoxSequence<2> s = box.sequence(); s.hasNext();) 
            send(&(*grid)[s.next()], dest, 1, datatype); 
    }

    /**
     * receives a sub-box of @a grid to @a dest asynchronously.
     */
    template<typename T>
    void recvGridBox(
        Grid<T>* grid, 
        const CoordBox<2>& box, 
        const int& src,
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        for (CoordBoxSequence<2> s = box.sequence(); s.hasNext();) 
            recv(&((*grid)[s.next()]), src, 1, datatype); 
    }

    template<typename T>
    void sendVec(
        const SuperVector<T> *vec, 
        const int& dest, 
        const int& waitTag = 0, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>())
    {
        MPI::Request req = comm->Isend(
            &(const_cast<UVec&>(*vec))[0], 
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

    template<int DIM>
    void sendRegion(const Region<DIM>& region, const int& dest)
    {
        unsigned numStreaks = region.numStreaks();
        MPI::Request req = comm->Isend(&numStreaks, 1, MPI::UNSIGNED, dest, tag);
        SuperVector<Streak<DIM> > buf = region.toVector();
        comm->Send(&buf[0], numStreaks, Typemaps::lookup<Streak<DIM> >(), dest, tag);
        req.Wait();
    }

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
        for (StreakIterator<DIM> i = region.beginStreak(); i != region.endStreak(); ++i) 
            recv(&(*stripe)[i->origin], src, i->length(), tag, datatype);
    }

    template<typename GRID_TYPE, int DIM>
    void sendUnregisteredRegion(GRID_TYPE *stripe, 
                                const Region<DIM>& region, 
                                const int& dest, 
                                const int& tag,
                                const MPI::Datatype& datatype)
    {
        for (StreakIterator<DIM> i = region.beginStreak(); i != region.endStreak(); ++i) 
            send(&(*stripe)[i->origin], dest, i->length(), tag, datatype);
    }
        
    template<typename T>
    MPIRegionPointer registerRegion(
        const T *base,
        const SuperVector<const T*>& addresses,
        const SuperVector<unsigned>& rawLengths,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        return registerRegion(
            base, 
            addresses.begin(), 
            rawLengths.begin(), 
            addresses.size(), 
            datatype);
    }

    template<typename T>
    MPIRegionPointer registerRegion(
        const T *base,
        const SuperVector<T*>& addresses,
        const SuperVector<unsigned>& rawLengths,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        return registerRegion(
            base, 
            addresses.begin(), 
            rawLengths.begin(), 
            addresses.size(), 
            datatype);
    }

    template<typename T, typename ITERATOR1, typename ITERATOR2>
    MPIRegionPointer registerRegion(
        const T *base,
        ITERATOR1 addresses,
        ITERATOR2 rawLengths,
        const unsigned& length,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        // empty regions would lead to illegal empty datatypes
        if (length == 0)
            return MPIRegionPointer();

        SuperVector<ChunkSpec> specs(length);        
        for (SuperVector<ChunkSpec>::iterator i = specs.begin(); i != specs.end(); ++i) {
            *i = ChunkSpec(*addresses, *rawLengths);
            ++addresses;
            ++rawLengths;
        }
        std::sort(&specs[0], &specs[specs.size()], addressLower);

        SuperVector<MPI::Aint> displacements(length);
        SuperVector<int> lengths(length);
        for (unsigned i = 0; i < length; i++) {
            displacements[i] = 
                MPI::Get_address(const_cast<void*>(specs[i].first)) - 
                MPI::Get_address(const_cast<T*>(base));
            lengths[i] = specs[i].second;
        }

        MPI::Datatype newType = datatype.Create_hindexed(
            length, &lengths[0], &displacements[0]);
        newType.Commit();
        MPIRegion *region = new MPIRegion;
        region->indices = newType;
        return boost::shared_ptr<MPIRegion>(region);
    }

    template<typename T, int DIM>
    MPIRegionPointer registerRegion(
        const Grid<T>& grid,
        const CoordBox<2>& box,
        const Coord<DIM>& base,
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        SuperVector<T*> disps;
        SuperVector<unsigned> lengths(box.dimensions.y(), box.dimensions.x());

        int x = box.origin.x();
        for (int y = box.origin.y(); 
             y < box.origin.y() + (int)box.dimensions.y(); y++)
            disps.push_back(const_cast<T*>(&(grid[Coord<DIM>(x, y)])));

        return registerRegion(
            const_cast<T*>(&(grid[base])), disps, lengths, datatype);
    }

    template<typename GRID_TYPE, int DIM>
    inline MPIRegionPointer registerRegion(
        const GRID_TYPE& grid, 
        const Region<DIM>& region) const
    {
        StreakToAddressTranslatingIterator<
            typename GRID_TYPE::CellType, 
            GRID_TYPE, 
            DIM> addresses(&grid, region.beginStreak());
        StreakToLengthTranslatingIterator<DIM> lengths(region.beginStreak());
        return registerRegion(
            grid.baseAddress(), addresses, lengths, region.numStreaks());
    }

    template<typename T>
    inline void sendRegion(
        const T *base,
        const MPIRegionPointer& region,
        const int& dest,
        const int& waitTag = 0)
    {
        // skip empty regions (nil pointers)
        if (!region)
            return;
        MPI::Request req = comm->Isend(base, 1, region->indices, dest, tag);
        requests[waitTag].push_back(req);
    }

    template<typename T>
    inline void recvRegion(
        T *base,
        const MPIRegionPointer& region,
        const int& src,
        const int& waitTag = 0)
    {
        // skip empty regions (nil pointers)
        if (!region)
            return;
        MPI::Request req = comm->Irecv(base, 1, region->indices, src, tag);
        requests[waitTag].push_back(req);    
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
        const T& source, 
        SuperVector<T> *target, 
        const MPI::Datatype& datatype = Typemaps::lookup<T>()) const
    {
        comm->Allgather(&source, 1, datatype, &(target->front()), 1, datatype);
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
        inline StreakToAddressTranslatingIterator(const GRID_TYPE *_grid, StreakIterator<DIM> _iter) :
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
        StreakIterator<DIM> iter;
    };

    template<int DIM>
    class StreakToLengthTranslatingIterator 
    {
    public:
        inline StreakToLengthTranslatingIterator(StreakIterator<DIM> _iter) :
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
        StreakIterator<DIM> iter;
    };

    static bool addressLower(ChunkSpec a, ChunkSpec b)
    {
        return a.first < b.first;
    }

};

}


#endif
#endif
