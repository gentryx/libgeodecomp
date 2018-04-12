#ifndef LIBGEODECOMP_PARALLELIZATION_SIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_SIMULATOR_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/soagrid.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * A Simulator controls the workflow of the simulation. It also needs
 * to interface with the Initializer (for setting up the initial
 * grid), Writer objects (for output) and Steerer objects (for input
 * at runtime). Simulator itself is just an abstract base class,
 * implementations may target different hardware architectures (e.g.
 * CUDASimulator or SerialSimulator).
 */
template<typename CELL_TYPE>
class Simulator
{
public:
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    static const int DIM = Topology::DIM;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;
    typedef GridBase<CELL_TYPE, DIM> GridType;
    typedef typename SharedPtr<Initializer<CELL_TYPE> >::Type InitPtr;
    typedef typename SharedPtr<Steerer<CELL_TYPE> >::Type SteererPtr;
    typedef std::vector<SteererPtr> SteererVector;

    /**
     * Creates the abstract Simulator object. The Initializer is
     * assumed to belong to the Simulator, which means that it'll
     * delete the  initializer at the end of its lifetime.
     */
    inline explicit Simulator(Initializer<CELL_TYPE> *initializer) :
        stepNum(0),
        initializer(initializer),
        gridDim(initializer->gridDimensions())
    {}

    inline explicit Simulator(const InitPtr& initializer) :
        stepNum(0),
        initializer(initializer),
        gridDim(initializer->gridDimensions())
    {}

    inline virtual ~Simulator()
    {}

    /**
     * performs a single simulation step.
     */
    virtual void step() = 0;

    /**
     * performs step() until the maximum number of steps is reached.
     */
    virtual void run() = 0;

    /**
     * returns the number of the current logical simulation step.
     */
    virtual unsigned getStep() const
    {
        return stepNum;
    }

    virtual InitPtr getInitializer() const
    {
        return initializer;
    }

    /**
     * adds a Steerer which will be called back before simulation
     * steps as specified by the Steerer's period. The Steerer is
     * assumed to be owned by the Simulator. It will destroy the
     * Steerer upon its death.
     */
    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        steerers << SteererPtr(steerer);
    }

    /**
     * Returns histograms which detail how much execution time was
     * spent on which part of the algorithm. Will return one element
     * per rank.
     */
    virtual std::vector<Chronometer> gatherStatistics() = 0;

protected:
    Chronometer chronometer;
    unsigned stepNum;
    InitPtr initializer;
    SteererVector steerers;
    Coord<DIM> gridDim;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
