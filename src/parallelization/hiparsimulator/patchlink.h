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
        inline Link(
            MPILayer *_mpiLayer,
            const int& _tag) :
            mpiLayer(_mpiLayer),
            tag(_tag)
        {}

        inline void setRegion(const Region<DIM>& newRegion)
        {
            region = newRegion;
        }

        inline void wait()
        {
            mpiLayer->wait(tag);
        }

    protected:
        MPILayer *mpiLayer;
        Region<DIM> region;
        int tag;
    };

    class Accepter : 
        public Link,
        public PatchAccepter<GRID_TYPE>
    {
    public:
        virtual void put(
            const GRID_TYPE& grid, 
            const Region<DIM>& /*validRegion*/, 
            const unsigned& nanoStep) 
        {
            if (!this->checkNanoStepPut(nanoStep))
                return;
        }
    };

    class Provider : 
        public Link,
        public PatchProvider<GRID_TYPE>
    {
        virtual void get(
            GRID_TYPE& destinationGrid, 
            const Region<DIM>& patchableRegion, 
            const unsigned& nanoStep,
            const bool& remove=true) 
        {
        }
    };

};

}
}
#endif
