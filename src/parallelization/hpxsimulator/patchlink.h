
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

            boost::shared_ptr<BufferType> buffer(new BufferType(region.size()));
            GridVecConv::gridToVector(grid, buffer.get(), region);

            hpx::wait(putFuture);
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
    public:
        Provider(const Region<DIM>& region) :
            Link(region),
            recvFuture(recvPromise.get_future())
        {}

        ~Provider()
        {
            if(!recvFuture.is_ready()) {
                recvPromise.set_value(-1);
            }
            if(!bufferFuture.is_ready()) {
                bufferPromise.set_value(boost::shared_ptr<BufferType>());
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
            boost::shared_ptr<BufferType> buffer = bufferFuture.get();

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
            storedNanoSteps << nanoStep;
            bufferPromise = hpx::lcos::local::promise<boost::shared_ptr<BufferType> >();
            bufferFuture = bufferPromise.get_future();
            BOOST_ASSERT(!recvPromise.is_ready());
            recvPromise.set_value(nanoStep);
        }

        void setBuffer(boost::shared_ptr<BufferType> buffer, long nanoStep)
        {
            hpx::wait(recvFuture);
            BOOST_ASSERT(recvFuture.get() == nanoStep);
            BOOST_ASSERT(!bufferPromise.is_ready());
            recvPromise = hpx::lcos::local::promise<long>();
            recvFuture = recvPromise.get_future();
            bufferPromise.set_value(buffer);
        }

    private:

        hpx::lcos::local::promise<boost::shared_ptr<BufferType> > bufferPromise;
        hpx::future<boost::shared_ptr<BufferType> > bufferFuture;
        hpx::lcos::local::promise<long> recvPromise;
        hpx::future<long> recvFuture;
    };
};

}
}

#endif
#endif
