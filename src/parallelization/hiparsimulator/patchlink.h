#ifndef _libgeodecomp_parallelization_hiparsimulator_patchlink_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchlink_h_

#include <deque>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * PatchLink encapsulates the transmission of patches to and from
 * remote processes. PatchLink::Accepter takes the patches from a
 * Stepper hands them on to MPI, while PatchLink::Provider will receive
 * the patches from the net and provide then to a Stepper.
 */
template<class GRID_TYPE>
class PatchLink
{
public:
    const static int DIM = GRID_TYPE::DIM;

    class Link 
    {
    public:
        typedef typename GRID_TYPE::CellType CellType;

        // fixme: patch link needs to be unique between the sending
        // and the receiving node. ensure this by e.g. a local
        // registry coupled with a peer exchange
        inline Link(
            const Region<DIM>& _region,
            const int& _tag) :
            lastNanoStep(0),
            stride(1),
            region(_region),
            buffer(_region.size()),
            tag(_tag)
        {}

        virtual ~Link()
        {
            this->wait();
        }

        virtual void charge(const long& next, const long& last, const long& newStride) 
        {          
            lastNanoStep = last;
            stride = newStride;
        }

        inline void wait()
        {
            mpiLayer.wait(tag);
        }

    protected:
        long lastNanoStep;
        long stride;
        MPILayer mpiLayer;
        Region<DIM> region;
        SuperVector<CellType> buffer;
        int tag;
    };

    class Accepter : 
        public Link,
        public PatchAccepter<GRID_TYPE>
    {
    public:
        inline Accepter(
            const Region<DIM>& _region=Region<DIM>(),
            const int& _dest=0,
            const int& _tag=0) :
            Link(_region, _tag),
            dest(_dest)
        {}

        virtual void charge(const long& next, const long& last, const long& newStride) 
        {
            Link::charge(next, last, newStride);
            this->pushRequest(next);
        }

        virtual void put(
            const GRID_TYPE& grid, 
            const Region<DIM>& /*validRegion*/, 
            const long& nanoStep) 
        {
            this->wait();
            if (!this->checkNanoStepPut(nanoStep))
                return;

            GridVecConv::gridToVector(grid, &this->buffer, this->region);
            this->mpiLayer.send(
                &this->buffer[0], dest, this->buffer.size(), this->tag);
            long nextNanoStep = this->requestedNanoSteps.min() + this->stride;
            if (nextNanoStep < this->lastNanoStep)
                this->requestedNanoSteps << nextNanoStep;
            this->requestedNanoSteps.erase_min();
        }

    private:
        int dest;
    };

    class Provider : 
        public Link,
        public PatchProvider<GRID_TYPE>
    {
    public:
        inline Provider(
            const Region<DIM>& _region=Region<DIM>(),
            const int& _source=0,
            const int& _tag=0) :
            Link(_region, _tag),
            source(_source)
        {}

        virtual void charge(const long& next, const long& last, const long& newStride) 
        {
            Link::charge(next, last, newStride);
            recv(next);
        }

        virtual void get(
            GRID_TYPE *grid, 
            const Region<DIM>& patchableRegion, 
            const long& nanoStep,
            const bool& remove=true) 
        {
            this->checkNanoStepGet(nanoStep);
            this->wait();
            GridVecConv::vectorToGrid(this->buffer, grid, this->region);

            long nextNanoStep = this->storedNanoSteps.min() + this->stride;
            if (nextNanoStep < this->lastNanoStep)
                recv(nextNanoStep);
            this->storedNanoSteps.erase_min();
        }

        void recv(const long& nanoStep)
        {
            this->storedNanoSteps << nanoStep;
            this->mpiLayer.recv(&this->buffer[0], source, this->buffer.size(), this->tag);
        }

    private:
        int source;
    };

};

}
}
#endif
