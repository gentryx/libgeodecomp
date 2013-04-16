#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_MPIIOINITIALIZER_H
#define LIBGEODECOMP_IO_MPIIOINITIALIZER_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/mpilayer/typemaps.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MPIIOInitializer : public Initializer<CELL_TYPE>
{    
public:
    static const int DIM = CELL_TYPE::Topology::DIM;

    MPIIOInitializer(
        const std::string& filename, 
        const MPI::Datatype& mpiDatatype = Typemaps::lookup<CELL_TYPE>(),
        const MPI::Intracomm& comm = MPI::COMM_WORLD) : 
        file(filename),
        datatype(mpiDatatype),
        communicator(comm)
    {
        MPIIO<CELL_TYPE>::readMetadata(
            &dimensions, &currentStep, &maximumSteps, file, communicator);
    }

    virtual void grid(GridBase<CELL_TYPE, DIM> *target) 
    {
        Region<DIM> region;
        region << target->boundingBox();
        MPIIO<CELL_TYPE>::readRegion(target, file, region, communicator, datatype);
    }

    virtual Coord<DIM> gridDimensions() const 
    {
        return dimensions;
    }

    virtual unsigned maxSteps() const 
    {
        return maximumSteps;
    }

    virtual unsigned startStep() const 
    {
        return currentStep;
    }

private:
    std::string file;
    MPI::Datatype datatype;
    MPI::Intracomm communicator;
    unsigned currentStep;
    unsigned maximumSteps;
    Coord<DIM> dimensions;
};

}

#endif
#endif
