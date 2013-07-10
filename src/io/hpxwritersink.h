
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_HPXWRITERSINK_H
#define LIBGEODECOMP_IO_HPXWRITERSINK_H

#include <libgeodecomp/io/hpxwritersinkserver.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/new.hpp>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

template<typename CELL_TYPE>
class HpxWriterSink
{
public:
    friend class boost::serialization::access;

    typedef HpxWriterSinkServer<CELL_TYPE> ComponentType;
    typedef typename CELL_TYPE::Topology Topology;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;
    typedef SuperVector<CELL_TYPE> BufferType;

    typedef
        hpx::components::server::create_component_action2<
            ComponentType,
            boost::shared_ptr<Writer<CELL_TYPE> >,
            std::size_t
        >
        ComponentWriterCreateActionType;

    typedef
        hpx::components::server::create_component_action2<
            ComponentType,
            boost::shared_ptr<ParallelWriter<CELL_TYPE> >,
            std::size_t
        >
        ComponentParallelWriterCreateActionType;

    HpxWriterSink() {}

    HpxWriterSink(
        ParallelWriter<CELL_TYPE> *parallelWriter,
        std::size_t numUpdateGroups
    )
    {
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer(parallelWriter);
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                writer,
                numUpdateGroups);
    }

    HpxWriterSink(
        Writer<CELL_TYPE> *serialWriter,
        std::size_t numUpdateGroups
    )
    {
        boost::shared_ptr<Writer<CELL_TYPE> > writer(serialWriter);
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                writer,
                numUpdateGroups);
    }

    HpxWriterSink(const HpxWriterSink& sink)
      : thisId(sink.thisId)
    {}

    void stepFinished(
        const GridType& grid,
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        BufferType buffer(validRegion.size());

        HiParSimulator::GridVecConv::gridToVector(grid, &buffer, validRegion);
        hpx::apply<typename ComponentType::StepFinishedAction>(
            thisId.get(),
            buffer,
            validRegion,
            globalDimensions,
            step,
            event,
            rank,
            lastCall
        );
    }

private:
    hpx::future<hpx::naming::id_type> thisId;

    template<typename ARCHIVE>
    void load(ARCHIVE& ar, unsigned)
    {
        hpx::naming::id_type id;
        ar & id;
        thisId = hpx::make_ready_future(id);
    }

    template<typename ARCHIVE>
    void save(ARCHIVE& ar, unsigned) const
    {
        hpx::naming::id_type id(thisId.get());
        ar & id;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

}

#endif
#endif
