
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H
#define LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/hpxwritersink.h>

namespace LibGeoDecomp {

template <typename CELL_TYPE>
class HpxWriterCollector : public ParallelWriter<CELL_TYPE>
{
public:
    friend class boost::serialization::access;

    typedef typename CELL_TYPE::Topology Topology;

    static const int DIM = CELL_TYPE::Topology::DIM;

    using ParallelWriter<CELL_TYPE>::period;
    using ParallelWriter<CELL_TYPE>::prefix;
    typedef typename ParallelWriter<CELL_TYPE>::GridType GridType;
    typedef typename ParallelWriter<CELL_TYPE>::RegionType RegionType;
    typedef typename ParallelWriter<CELL_TYPE>::CoordType CoordType;
    
    HpxWriterCollector() {}

    HpxWriterCollector(
        unsigned period,
        HpxWriterSink<CELL_TYPE> const & sink) :
        ParallelWriter<CELL_TYPE>("", period),
        sink(sink)
    {
    }

    ParallelWriter<CELL_TYPE> * clone()
    {
        return new HpxWriterCollector(period, sink);
    }
    
    void stepFinished(
        const GridType& grid,
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        sink.stepFinished(grid, validRegion, globalDimensions, step, event, rank, lastCall);
    }

private:
    HpxWriterSink<CELL_TYPE> sink;

    template <class ARCHIVE>
    void serialize(ARCHIVE & ar, unsigned)
    {
        ar & boost::serialization::base_object<ParallelWriter<CELL_TYPE> >(*this);
        ar & sink;
    }
};

}

#endif
#endif
