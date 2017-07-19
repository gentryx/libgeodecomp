#ifndef LIBGEODECOMP_COMMUNICATION_HPXPATCHLINK_H
#define LIBGEODECOMP_COMMUNICATION_HPXPATCHLINK_H

#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

namespace LibGeoDecomp {

template <class GRID_TYPE>
class HPXPatchLink
{
public:
    friend class PatchLinkTest;

    typedef typename GRID_TYPE::CellType CellType;
    typedef typename SerializationBuffer<CellType>::BufferType BufferType;
    typedef typename SerializationBuffer<CellType>::FixedSize FixedSize;

    const static int DIM = GRID_TYPE::DIM;

    class Link
    {
    public:
        static std::string genLinkName(const std::string& basename, std::size_t sourceRank, std::size_t targetRank)
        {
            return basename + "/PatchLink/" +
                StringOps::itoa(sourceRank) + "-" + StringOps::itoa(targetRank);
        }

        /**
         * We'll use the linkName to uniquely identify the endpoints in AGAS.
         */
        inline Link(
            const Region<DIM>& region,
            const std::string& linkName) :
            linkName(linkName),
            lastNanoStep(0),
            stride(1),
            region(region)
        {}

        virtual ~Link()
        {}

        /**
         * Should be called prior to destruction to allow
         * implementations to perform any cleanup actions (e.g. to
         * post any receives to pending transmissions).
         */
        virtual void cleanup()
        {}

        virtual void charge(std::size_t next, std::size_t last, std::size_t newStride)
        {
            lastNanoStep = last;
            stride = newStride;
        }

    protected:
        std::string linkName;
        std::size_t lastNanoStep;
        long stride;
        Region<DIM> region;
        BufferType buffer;
    };

    class Accepter :
        public Link,
        public PatchAccepter<GRID_TYPE>
    {
    public:
        using Link::linkName;
        using Link::buffer;
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
        using PatchAccepter<GRID_TYPE>::infinity;
        using PatchAccepter<GRID_TYPE>::pushRequest;
        using PatchAccepter<GRID_TYPE>::requestedNanoSteps;

        inline Accepter(
            const Region<DIM>& region,
            std::string basename,
            const std::size_t source,
            const std::size_t target) :
            Link(region, Link::genLinkName(basename, source, target)),
            receiverID(HPXReceiver<BufferType>::find(linkName).get())
        {
            // FIXME: this shouldn't be needed. However, the apply in the put function
            // currently leads to unnecessary decref/incref requests which limits
            // scalability. This is a performance bug in HPX and needs to be fixed
            // eventually
            receiverID.make_unmanaged();
        }

        virtual void charge(std::size_t next, std::size_t last, std::size_t newStride)
        {
            Link::charge(next, last, newStride);
            pushRequest(next);
        }

        virtual void put(
            const GRID_TYPE& grid,
            const Region<DIM>& /*validRegion*/,
            const Coord<DIM>& globalGridDimensions,
            const std::size_t nanoStep,
            const std::size_t rank)
        {
            if (!checkNanoStepPut(nanoStep)) {
                return;
            }

            buffer = SerializationBuffer<CellType>::create(region);
            grid.saveRegion(&buffer, region);
            hpx::apply(typename HPXReceiver<BufferType>::receiveAction(), receiverID,  nanoStep, std::move(buffer));

            std::size_t nextNanoStep = (min)(requestedNanoSteps) + stride;
            if ((lastNanoStep == infinity()) ||
                (nextNanoStep < lastNanoStep)) {
                requestedNanoSteps << nextNanoStep;
            }

            erase_min(requestedNanoSteps);
        }

    private:
        hpx::id_type receiverID;
    };

    class Provider :
        public Link,
        public PatchProvider<GRID_TYPE>
    {
    public:
        using Link::linkName;
        using Link::buffer;
        using Link::lastNanoStep;
        using Link::region;
        using Link::stride;
        using PatchProvider<GRID_TYPE>::checkNanoStepGet;
        using PatchProvider<GRID_TYPE>::infinity;
        using PatchProvider<GRID_TYPE>::storedNanoSteps;

        inline
        Provider(
            const Region<DIM>& region,
            std::string basename,
            const std::size_t source,
            const std::size_t target) :
            Link(region, Link::genLinkName(basename, source, target)),
            receiver(HPXReceiver<BufferType>::make(linkName).get())
        {}

        virtual void charge(std::size_t next, std::size_t last, std::size_t newStride)
        {
            Link::charge(next, last, newStride);
            recv(next);
        }

        virtual hpx::future<void> getAsync(
            GRID_TYPE *grid,
            const Region<DIM>& patchableRegion,
            const Coord<DIM>& globalGridDimensions,
            const std::size_t nanoStep,
            const std::size_t rank,
            const bool remove = true)
        {
            if (storedNanoSteps.empty() || (nanoStep < (min)(storedNanoSteps))) {
                return hpx::make_ready_future();
            }

            checkNanoStepGet(nanoStep);

            return receiver->get(nanoStep).then(
                [grid, this](hpx::future<BufferType> f) -> void {
                    grid->loadRegion(f.get(), region);

                    std::size_t nextNanoStep = (min)(storedNanoSteps) + stride;
                    if ((lastNanoStep == infinity()) ||
                        (nextNanoStep < lastNanoStep)) {
                        recv(nextNanoStep);
                    }

                    erase_min(storedNanoSteps);
                }
            );
        }

        virtual void get(
            GRID_TYPE *grid,
            const Region<DIM>& patchableRegion,
            const Coord<DIM>& globalGridDimensions,
            const std::size_t nanoStep,
            const std::size_t rank,
            const bool remove = true)
        {
            getAsync(grid, patchableRegion, globalGridDimensions, nanoStep, rank, remove).get();
        }

        void recv(const std::size_t nanoStep)
        {
            storedNanoSteps << nanoStep;
        }

    private:
        std::shared_ptr<HPXReceiver<BufferType> > receiver;
    };
};

}

#endif
