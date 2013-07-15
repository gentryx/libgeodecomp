
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
        hpx::components::server::create_component_action1<
            ComponentType,
            std::size_t
        >
        ComponentCreateActionType;

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

    HpxWriterSink() :
        stepFinishedFuture(hpx::make_ready_future())
    {}

    HpxWriterSink(const std::string& name) :
        thisId(hpx::naming::invalid_id),
        stepFinishedFuture(hpx::make_ready_future())
    {
        std::size_t retry = 0;

        hpx::naming::id_type id = hpx::naming::invalid_id;
        while(id == hpx::naming::invalid_id)
        {
            hpx::agas::resolve_name(name, id);
            if(retry > 10)
            {
                throw std::logic_error("Can't find the Writer Sink name");
            }
            hpx::this_thread::suspend();
            ++retry;
        }
        thisId = hpx::lcos::make_ready_future(id);
    }

    HpxWriterSink(
        unsigned period,
        std::size_t numUpdateGroups,
        const std::string& name) :
        period(period),
        stepFinishedFuture(hpx::make_ready_future())
    {
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                numUpdateGroups);
        hpx::agas::register_name(name, thisId.get());
    }

    HpxWriterSink(
        ParallelWriter<CELL_TYPE> *parallelWriter,
        std::size_t numUpdateGroups,
        const std::string& name = "") :
        period(parallelWriter->getPeriod()),
        stepFinishedFuture(hpx::make_ready_future())
    {
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer(parallelWriter);
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                writer,
                numUpdateGroups);
        if(name != "")
        {
            hpx::agas::register_name(name, thisId.get());
        }
    }

    HpxWriterSink(
        Writer<CELL_TYPE> *serialWriter,
        std::size_t numUpdateGroups,
        const std::string& name = "") :
        period(serialWriter->getPeriod()),
        stepFinishedFuture(hpx::make_ready_future())
    {
        boost::shared_ptr<Writer<CELL_TYPE> > writer(serialWriter);
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                writer,
                numUpdateGroups);
        if(name != "")
        {
            hpx::agas::register_name(name, thisId.get());
        }
    }

    HpxWriterSink(const HpxWriterSink& sink) :
        thisId(sink.thisId),
        period(sink.period),
        stepFinishedFuture(hpx::make_ready_future())
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
        hpx::wait(stepFinishedFuture);
        stepFinishedFuture = hpx::async<typename ComponentType::StepFinishedAction>(
        //hpx::apply<typename ComponentType::StepFinishedAction>(
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

    hpx::future<std::size_t> connectWriter(ParallelWriter<CELL_TYPE> *parallelWriter)
    {
        boost::shared_ptr<Writer<CELL_TYPE> > writer(parallelWriter);
        return
            hpx::async<typename ComponentType::ConnectParallelWriterAction>(
                thisId,
                writer);
    }

    hpx::future<std::size_t> connectWriter(Writer<CELL_TYPE> *serialWriter)
    {
        boost::shared_ptr<Writer<CELL_TYPE> > writer(serialWriter);
        return
            hpx::async<typename ComponentType::ConnectSerialWriterAction>(
                thisId,
                writer);
    }

    void disconnectWriter(std::size_t id)
    {
        typename ComponentType::DisconnectWriterAction()(thisId, id);
    }

    std::size_t getPeriod() const
    {
        return period;
    }

private:
    hpx::future<hpx::naming::id_type> thisId;
    std::size_t period;
    hpx::future<void> stepFinishedFuture;

    template<typename ARCHIVE>
    void load(ARCHIVE& ar, unsigned)
    {
        hpx::naming::id_type id;
        ar & id;
        ar & period;
        thisId = hpx::make_ready_future(id);
    }

    template<typename ARCHIVE>
    void save(ARCHIVE& ar, unsigned) const
    {
        hpx::naming::id_type id(thisId.get());
        ar & id;
        ar & period;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

}

#endif
#endif
