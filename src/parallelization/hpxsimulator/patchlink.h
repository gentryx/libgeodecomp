
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXPATCHLINKS_H
#define LIBGEODECOMP_PARALLELIZATION_HPXPATCHLINKS_H

#include <libgeodecomp/storage/gridvecconv.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/local/packaged_task.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class GRID_TYPE, class UPDATE_GROUP>
class PatchLink
{
public:
    const static int DIM = GRID_TYPE::DIM;
    const static int ENDLESS = -1;

    typedef typename GRID_TYPE::CellType CellType;
    typedef std::vector<CellType> BufferType;

    class Link
    {
    public:
        Link(const Region<DIM>& region) :
            lastNanoStep(0),
            stride(1),
            region(region)
        {}

        void charge(std::size_t last, long newStride)
        {
            lastNanoStep = last;
            stride = newStride;
        }

        virtual ~Link() {}

    protected:
        std::size_t lastNanoStep;
        long stride;
        Region<DIM> region;
    };

    class Accepter :
        public Link,
        public PatchAccepter<GRID_TYPE>
    {
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
        using PatchAccepter<GRID_TYPE>::pushRequest;
        using PatchAccepter<GRID_TYPE>::requestedNanoSteps;
    public:

        Accepter(
            const Region<DIM>& region,
            std::size_t rank,
            const UPDATE_GROUP& dest) :
            Link(region),
            rank(rank),
            dest(dest),
            putFuture(hpx::make_ready_future())
        {}

        void charge(std::size_t next, std::size_t last, std::size_t newStride)
        {
            Link::charge(last, newStride);
            pushRequest(next);
        }

        void put(
            const GRID_TYPE& grid,
            const Region<DIM>&, /*validRegion*/
            const std::size_t nanoStep)
        {
            if (!this->checkNanoStepPut(nanoStep)) {
                return;
            }

            hpx::wait(putFuture);
            boost::shared_ptr<BufferType> buffer(new BufferType(region.size()));
            GridVecConv::gridToVector(grid, buffer.get(), region);

            putFuture = dest.setOuterGhostZone(rank, buffer, nanoStep);

            std::size_t nextNanoStep = (min)(requestedNanoSteps) + stride;
            if ((lastNanoStep == std::size_t(-1)) ||
               (nextNanoStep < lastNanoStep)) {
                requestedNanoSteps << nextNanoStep;
            }
            erase_min(requestedNanoSteps);
        }

    private:
        std::size_t rank;
        UPDATE_GROUP dest;
        hpx::future<void> putFuture;
    };

    class Provider :
        public Link,
        public PatchProvider<GRID_TYPE>
    {
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using PatchProvider<GRID_TYPE>::checkNanoStepGet;
        using PatchProvider<GRID_TYPE>::storedNanoSteps;

        typedef hpx::lcos::local::spinlock MutexType;

        struct Receiver
        {
            Receiver()
              : bufferFuture(bufferPromise.get_future())
            {}

            hpx::lcos::local::promise<boost::shared_ptr<BufferType> > bufferPromise;
            hpx::lcos::future<boost::shared_ptr<BufferType> > bufferFuture;
        };

        typedef std::map<std::size_t, Receiver> ReceiverMap;

    public:
        Provider(const Region<DIM>& region) :
            Link(region)/*,
            recvFuture(recvPromise.get_future())*/
        {}

        ~Provider()
        {
            /*
            if(!recvFuture.is_ready()) {
                recvPromise.set_value(-1);
            }
            if(!bufferFuture.is_ready()) {
                bufferPromise.set_value(boost::shared_ptr<BufferType>());
            }
            */
        }

        void charge(long next, long last, long newStride)
        {
            Link::charge(last, newStride);
            stride = newStride;
            recv(next);
        }

        void get(
            GRID_TYPE *grid,
            const Region<DIM>&,
            const std::size_t nanoStep,
            const bool remove=true
        )
        {
            throw std::logic_error("not implemented!");
        }

        void get(
            GRID_TYPE *grid,
            const Region<DIM>&,
            const std::size_t nanoStep,
            hpx::lcos::local::spinlock& gridMutex,
            const bool remove=true
        )
        {
            //boost::shared_ptr<BufferType> buffer = bufferFuture.get();
            //std::cout << "get ... " << nanoStep << "\n";
            boost::shared_ptr<BufferType> buffer = getBuffer(nanoStep);

            checkNanoStepGet(nanoStep);
            {
                //hpx::lcos::local::spinlock::scoped_lock lock(gridMutex);
                GridVecConv::vectorToGrid(*buffer, grid, region);
            }

            std::size_t nextNanoStep = (min)(storedNanoSteps) + stride;
            if ((lastNanoStep == std::size_t(-1)) ||
               (nextNanoStep < lastNanoStep)) {
                recv(nextNanoStep);
            }

            erase_min(storedNanoSteps);
        }

        void recv(long nanoStep)
        {
            MutexType::scoped_lock lock(mutex);
            storedNanoSteps << nanoStep;
            //createReceiver(nanoStep);
            /*
            bufferPromise = hpx::lcos::local::promise<boost::shared_ptr<BufferType> >();
            bufferFuture = bufferPromise.get_future();
            BOOST_ASSERT(!recvPromise.is_ready());
            recvPromise.set_value(nanoStep);
            */
        }

        void setBuffer(boost::shared_ptr<BufferType> buffer, long nanoStep)
        {
            //std::cout << "set ... " << nanoStep << "\n";
            MutexType::scoped_lock lock(mutex);
            typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
            if(it == receiverMap.end()) {
                it = createReceiver(nanoStep);
            }
            it->second.bufferPromise.set_value(buffer);
            //receiverMap.erase(it);

            /*
            hpx::wait(recvFuture);
            {
                MutexType::scoped_lock lock(mutex);
                BOOST_ASSERT(recvFuture.get() == nanoStep);
                BOOST_ASSERT(!bufferPromise.is_ready());
                recvPromise = hpx::lcos::local::promise<long>();
                recvFuture = recvPromise.get_future();
                bufferPromise.set_value(buffer);
            }
            */
        }

    private:

        boost::shared_ptr<BufferType> getBuffer(std::size_t nanoStep)
        {
            hpx::future<boost::shared_ptr<BufferType> > resFuture;
            {
                MutexType::scoped_lock lock(mutex);
                typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
                if(it == receiverMap.end()) {
                    it = createReceiver(nanoStep);
                }
                resFuture = it->second.bufferFuture;
            }
            boost::shared_ptr<BufferType> res = resFuture.get();
            {
                MutexType::scoped_lock lock(mutex);
                typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
                if(it == receiverMap.end()) {
                    std::cerr << "attempt to erase non existing receiver " << nanoStep << "\n";
                    throw "";
                }
                receiverMap.erase(it);
            }
            return res;
        }

        typename ReceiverMap::iterator createReceiver(std::size_t nanoStep)
        {
            std::pair<typename ReceiverMap::iterator, bool> createRes
                = receiverMap.insert(std::make_pair(nanoStep, Receiver()));
            if(createRes.second == false) {
                std::cerr << "nano step " << nanoStep << " already inserted\n";
                throw "";
            }
            return createRes.first;
        }

        MutexType mutex;
        ReceiverMap receiverMap;
        /*
        hpx::lcos::local::promise<boost::shared_ptr<BufferType> > bufferPromise;
        hpx::future<boost::shared_ptr<BufferType> > bufferFuture;
        hpx::lcos::local::promise<long> recvPromise;
        hpx::future<long> recvFuture;
        */
    };
};

}
}

#endif
#endif
