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
        if ((this->sim->getStep() % this->period) == 0) 
            writeGrid();
    }

    virtual void allDone()
    {
        writeGrid();
    }

private:
    MPI::Intracomm comm;
    MPI::Datatype datatype;

    std::string filename(const unsigned& step) const 
    {
        std::ostringstream buf;
        buf << this->prefix << std::setfill('0') << std::setw(5) << step << ".mpiio";
        return buf.str();
    }

    void writeGrid()
    {
        Region<DIM> region;
        CoordBox<DIM> boundingBox = this->sim->getInitializer()->gridBox();
        region << boundingBox;
        unsigned step = this->sim->getStep();
        
        MPIIO<CELL_TYPE>::writeRegion(
            *this->sim->getGrid(), 
            boundingBox.dimensions,
            step,
            this->sim->getInitializer()->maxSteps(),
            this->filename(step),
            region,
            this->datatype,
            this->comm);
    }
};

}

#endif
#endif
