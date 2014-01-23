#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_PATCHLINK_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_PATCHLINK_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/storage/gridvecconv.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/local/packaged_task.hpp>

#include <boost/foreach.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class GRID_TYPE, class UPDATE_GROUP>
class PatchLink
{
public:
    const static int DIM = GRID_TYPE::DIM;

    typedef typename GRID_TYPE::CellType CellType;
    typedef std::vector<CellType> BufferType;

    static inline std::size_t infinity()
    {
        return std::numeric_limits<std::size_t>::max();
    }

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
            const UPDATE_GROUP& ugDest) :
            Link(region),
            rank(rank),
            putFuture(hpx::make_ready_future())
        {
            // We use an unmanaged GID here to avoid unnecessary credit splits
            // we know that we keep a reference in the update group anyways.
            hpx::naming::gid_type gid = ugDest.gid().get_gid();
            hpx::naming::detail::strip_credit_from_gid(gid);
            dest = UPDATE_GROUP(hpx::id_type(gid, hpx::id_type::unmanaged));
        }

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
        hpx::unique_future<void> putFuture;
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

        typedef
            boost::shared_ptr<hpx::lcos::local::promise<boost::shared_ptr<BufferType> > >
            Receiver;

        typedef std::map<std::size_t, Receiver> ReceiverMap;

    public:
        Provider(const Region<DIM>& region) :
            Link(region)
        {}

        ~Provider()
        {
            BOOST_FOREACH(typename ReceiverMap::value_type & recv, receiverMap)
            {
                if(recv.second->valid())
                {
                    recv.second->get_future();
                }
            }
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
            boost::shared_ptr<BufferType> buffer = getBuffer(nanoStep);

            checkNanoStepGet(nanoStep);
            {
                hpx::lcos::local::spinlock::scoped_lock lock(gridMutex);
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
        }

        void setBuffer(boost::shared_ptr<BufferType> buffer, long nanoStep)
        {
            MutexType::scoped_lock lock(mutex);
            typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
            if(it == receiverMap.end()) {
                it = createReceiver(nanoStep);
            }
            it->second->set_value(buffer);
        }

    private:

        boost::shared_ptr<BufferType> getBuffer(std::size_t nanoStep)
        {
            hpx::unique_future<boost::shared_ptr<BufferType> > resFuture;
            {
                MutexType::scoped_lock lock(mutex);
                typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
                if(it == receiverMap.end()) {
                    it = createReceiver(nanoStep);
                }
                resFuture = it->second->get_future();
            }
            boost::shared_ptr<BufferType> res = resFuture.get();
            {
                MutexType::scoped_lock lock(mutex);
                typename ReceiverMap::iterator it = receiverMap.find(nanoStep);
                if(it == receiverMap.end()) {
                    throw std::logic_error("attempt to erase non existing receiver");
                }
                receiverMap.erase(it);
            }
            return res;
        }

        typename ReceiverMap::iterator createReceiver(std::size_t nanoStep)
        {
            std::pair<typename ReceiverMap::iterator, bool> createRes
                = receiverMap.insert(
                    std::pair<std::size_t, Receiver>(
                        nanoStep,
                        boost::make_shared<hpx::lcos::local::promise<boost::shared_ptr<BufferType> > >()));
            if(createRes.second == false) {
                throw std::logic_error("nano step already inserted");
            }
            return createRes.first;
        }

        MutexType mutex;
        ReceiverMap receiverMap;
    };
};

}
}

#endif
#endif
