
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXPATCHLINKS_H
#define LIBGEODECOMP_PARALLELIZATION_HPXPATCHLINKS_H

#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

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
    typedef SuperVector<CellType> BufferType;

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

    protected:
        std::size_t lastNanoStep;
        long stride;
        Region<DIM> region;
    };

    class Accepter
      : public Link,
        public HiParSimulator::PatchAccepter<GRID_TYPE>
    {
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using HiParSimulator::PatchAccepter<GRID_TYPE>::checkNanoStepPut;
        using HiParSimulator::PatchAccepter<GRID_TYPE>::pushRequest;
        using HiParSimulator::PatchAccepter<GRID_TYPE>::requestedNanoSteps;
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
            HiParSimulator::GridVecConv::gridToVector(grid, buffer.get(), region);

            hpx::wait(putFuture);
            putFuture = dest.setOuterGhostZone(rank, buffer, nanoStep);

            std::size_t nextNanoStep = (requestedNanoSteps.min)() + stride;
            if ((lastNanoStep == std::size_t(-1)) ||
               (nextNanoStep < lastNanoStep)) {
                requestedNanoSteps << nextNanoStep;
            }
            requestedNanoSteps.erase_min();
        }

    private:
        std::size_t rank;
        UPDATE_GROUP dest;
        hpx::future<void> putFuture;
    };

    class Provider
      : public Link,
        public HiParSimulator::PatchProvider<GRID_TYPE>
    {
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using HiParSimulator::PatchProvider<GRID_TYPE>::checkNanoStepGet;
        using HiParSimulator::PatchProvider<GRID_TYPE>::storedNanoSteps;
    public:
        Provider(const Region<DIM>& region) :
            Link(region),
            recvFuture(recvPromise.get_future())
        {}

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
            boost::shared_ptr<BufferType> buffer = bufferFuture.get();

            checkNanoStepGet(nanoStep);
            HiParSimulator::GridVecConv::vectorToGrid(*buffer, grid, region);

            std::size_t nextNanoStep = (storedNanoSteps.min)() + stride;
            if ((lastNanoStep == std::size_t(-1)) ||
               (nextNanoStep < lastNanoStep)) {
                recv(nextNanoStep);
            }
            storedNanoSteps.erase_min();
        }

        void recv(long nanoStep)
        {
            storedNanoSteps << nanoStep;
            bufferPromise = hpx::lcos::local::promise<boost::shared_ptr<BufferType> >();
            bufferFuture = bufferPromise.get_future();
            BOOST_ASSERT(!recvPromise.ready());
            recvPromise.set_value(nanoStep);
        }

        void setBuffer(boost::shared_ptr<BufferType> buffer, long nanoStep)
        {
            hpx::wait(recvFuture);
            BOOST_ASSERT(recvFuture.get() == nanoStep);
            BOOST_ASSERT(!bufferPromise.ready());
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
