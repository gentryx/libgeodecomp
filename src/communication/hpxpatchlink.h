#ifndef LIBGEODECOMP_COMMUNICATION_HPXPATCHLINK_H
#define LIBGEODECOMP_COMMUNICATION_HPXPATCHLINK_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/storage/gridvecconv.h>
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
        /**
         * We'll use the linkName to uniquely identify the endpoints in AGAS.
         */
        inline Link(
            const Region<DIM>& region,
            const std::string& linkName) :
            linkName(linkName),
            lastNanoStep(0),
            stride(1),
            region(region),
            buffer(SerializationBuffer<CellType>::create(region))
        {}

        virtual ~Link()
        {
            wait();
        }

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

        inline void wait()
        {
            // fixme
            // mpiLayer.wait(tag);
        }

        inline void cancel()
        {
            // fixme
            // mpiLayer.cancelAll();
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
        using Link::wait;
        using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
        using PatchAccepter<GRID_TYPE>::infinity;
        using PatchAccepter<GRID_TYPE>::pushRequest;
        using PatchAccepter<GRID_TYPE>::requestedNanoSteps;

        inline Accepter(
            const Region<DIM>& region,
            std::string basename,
            const std::size_t source,
            const std::size_t target) :
            Link(region, Link::genLinkName(basename, source, target))
        {}

        virtual void charge(std::size_t next, std::size_t last, std::size_t newStride)
        {
            Link::charge(next, last, newStride);
            pushRequest(next);
        }

        virtual void put(
            const GRID_TYPE& grid,
            const Region<DIM>& /*validRegion*/,
            const std::size_t nanoStep)
        {
            if (!checkNanoStepPut(nanoStep)) {
                return;
            }

            // fixme
            // wait();
            // GridVecConv::gridToVector(grid, &buffer, region);
            // sendHeader(FixedSize());
            // mpiLayer.send(&buffer[0], dest, buffer.size(), tag, cellMPIDatatype);

            // std::size_t nextNanoStep = (min)(requestedNanoSteps) + stride;
            // if ((lastNanoStep == infinity()) ||
            //     (nextNanoStep < lastNanoStep)) {
            //     requestedNanoSteps << nextNanoStep;
            // }

            // erase_min(requestedNanoSteps);
        }
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
        using Link::tag;
        using Link::wait;
        using PatchProvider<GRID_TYPE>::checkNanoStepGet;
        using PatchProvider<GRID_TYPE>::infinity;
        using PatchProvider<GRID_TYPE>::storedNanoSteps;

        inline
        Provider(
            const Region<DIM>& region,
            std::string basename,
            const std::size_t source,
            const std::size_t target) :
            Link(region, Link::genLinkName(basename, source, target))
        {}

        virtual void cleanup()
        {
            // fixme
            // if (transmissionInFlight) {
            //     recvSecondPart(FixedSize());
            // }
        }

        virtual void charge(const std::size_t next, const std::size_t last, const std::size_t newStride)
        {
            Link::charge(next, last, newStride);
            // fixme
            // recv(next);
        }

        virtual void get(
            GRID_TYPE *grid,
            const Region<DIM>& patchableRegion,
            const std::size_t nanoStep,
            const bool remove = true)
        {
            if (storedNanoSteps.empty() || (nanoStep < (min)(storedNanoSteps))) {
                return;
            }

            checkNanoStepGet(nanoStep);
            // fixme
            // wait();
            // recvSecondPart(FixedSize());
            // transmissionInFlight = false;

            // GridVecConv::vectorToGrid(buffer, grid, region);

            // std::size_t nextNanoStep = (min)(storedNanoSteps) + stride;
            // if ((lastNanoStep == infinity()) ||
            //     (nextNanoStep < lastNanoStep)) {
            //     recv(nextNanoStep);
            // }

            // erase_min(storedNanoSteps);
        }

        void recv(const std::size_t nanoStep)
        {
            storedNanoSteps << nanoStep;
            // fixme:
            // recvFirstPart(FixedSize());
            // transmissionInFlight = true;
        }

    private:
        // bool transmissionInFlight;
    };
};

}

#endif

#endif
