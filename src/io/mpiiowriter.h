#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_mpiiowriter_h_
#define _libgeodecomp_io_mpiiowriter_h_

#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/mpilayer/typemaps.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MPIIOWriter : public Writer<CELL_TYPE>
{    
public:
    friend class MPIIOWriterTest;
    friend class MPIIOInitializerTest;

    static const int DIM = CELL_TYPE::Topology::DIMENSIONS;

    MPIIOWriter(
        const std::string& prefix, 
        MonolithicSimulator<CELL_TYPE> *sim, 
        const unsigned& period, 
        const MPI::Intracomm& communicator = MPI::COMM_WORLD,
        MPI::Datatype mpiDatatype = Typemaps::lookup<CELL_TYPE>()) :
        Writer<CELL_TYPE>(prefix, sim, period),
        comm(communicator),
        datatype(mpiDatatype)
    {}

    virtual void initialized() 
    {
        writeGrid();
    }

    virtual void stepFinished()
    {
        if ((sim->getStep() % period) == 0) 
            writeGrid();
    }

    virtual void allDone()
    {
        writeGrid();
    }

private:
    using Writer<CELL_TYPE>::sim;
    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    MPI::Intracomm comm;
    MPI::Datatype datatype;

    std::string filename(const unsigned& step) const 
    {
        std::ostringstream buf;
        buf << prefix << std::setfill('0') << std::setw(5) << step << ".mpiio";
        return buf.str();
    }

    void writeGrid()
    {
        Region<DIM> region;
        CoordBox<DIM> boundingBox = sim->getInitializer()->gridBox();
        region << boundingBox;
        unsigned step = sim->getStep();
        
        MPIIO<CELL_TYPE>::writeRegion(
            *sim->getGrid(), 
            boundingBox.dimensions,
            step,
            sim->getInitializer()->maxSteps(),
            filename(step),
            region,
            datatype,
            comm);
    }
};

}

#endif
#endif
