
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_HPXWRITERSINK_H
#define LIBGEODECOMP_IO_HPXWRITERSINK_H

#include <libgeodecomp/io/hpxwritersinkserver.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/wait_any.hpp>
#include <hpx/runtime/components/new.hpp>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

template<typename CELL_TYPE, typename CONVERTER = IdentityConverter<CELL_TYPE> >
class HpxWriterSink
{
public:
    friend class boost::serialization::access;

    typedef HpxWriterSinkServer<CELL_TYPE, CONVERTER> ComponentType;
    typedef typename CONVERTER::CellType CellType;
    typedef typename CellType::Topology Topology;
    typedef typename DistributedSimulator<CellType>::GridType GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;
    typedef SuperVector<CellType> BufferType;
    typedef hpx::lcos::local::spinlock MutexType;

    typedef
        hpx::components::server::create_component_action1<
            ComponentType,
            std::size_t
        >
        ComponentCreateActionType;

    typedef
        hpx::components::server::create_component_action2<
            ComponentType,
            boost::shared_ptr<Writer<CellType> >,
            std::size_t
        >
        ComponentWriterCreateActionType;

    typedef
        hpx::components::server::create_component_action2<
            ComponentType,
            boost::shared_ptr<ParallelWriter<CellType> >,
            std::size_t
        >
        ComponentParallelWriterCreateActionType;

    HpxWriterSink()
    {}

    HpxWriterSink(const std::string& name) :
        thisId(hpx::naming::invalid_id)
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
        period(period)
    {
        thisId
            = hpx::components::new_<ComponentType>(
                hpx::find_here(),
                numUpdateGroups);
        hpx::agas::register_name(name, thisId.get());
    }

    HpxWriterSink(
        ParallelWriter<CellType> *parallelWriter,
        std::size_t numUpdateGroups,
        const std::string& name = "") :
        period(parallelWriter->getPeriod())
    {
        boost::shared_ptr<ParallelWriter<CellType> > writer(parallelWriter);
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
        Writer<CellType> *serialWriter,
        std::size_t numUpdateGroups,
        const std::string& name = "") :
        period(serialWriter->getPeriod())
    {
        boost::shared_ptr<Writer<CellType> > writer(serialWriter);
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
        period(sink.period)
    {}

    void stepFinished(
        const typename DistributedSimulator<CELL_TYPE>::GridType& grid,
        const Region<CELL_TYPE::Topology::DIM>& validRegion,
        const Coord<CELL_TYPE::Topology::DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        boost::shared_ptr<BufferType> buffer(new BufferType(validRegion.size()));

        CellType *dest = &(*buffer)[0];

        for (typename Region<CELL_TYPE::Topology::DIM>::Iterator i = validRegion.begin();
             i != validRegion.end(); ++i) {
            *dest = CONVERTER()(grid.at(*i), globalDimensions, step, rank);
            ++dest;
        }

        hpx::future<void> stepFinishedFuture
            = hpx::async<typename ComponentType::StepFinishedAction>(
                thisId.get(),
                buffer,
                validRegion,
                globalDimensions,
                step,
                event,
                rank,
                lastCall
            );

        if(stepFinishedFutures.size() > 100) {
            MutexType::scoped_lock lk(mtx);
            HPX_STD_TUPLE<int, hpx::future<void> > res
                = hpx::wait_any(stepFinishedFutures);
            stepFinishedFutures[HPX_STD_GET(0, res)] = stepFinishedFuture;
        }
        else {
            MutexType::scoped_lock lk(mtx);
            stepFinishedFutures.push_back(stepFinishedFuture);
        }
    }

    hpx::future<std::size_t> connectWriter(ParallelWriter<CellType> *parallelWriter)
    {
        boost::shared_ptr<ParallelWriter<CellType> > writer(parallelWriter);
        return
            hpx::async<typename ComponentType::ConnectParallelWriterAction>(
                thisId.get(),
                writer);
    }

    hpx::future<std::size_t> connectWriter(Writer<CellType> *serialWriter)
    {
        boost::shared_ptr<Writer<CellType> > writer(serialWriter);
        return
            hpx::async<typename ComponentType::ConnectSerialWriterAction>(
                thisId.get(),
                writer);
    }

    void disconnectWriter(std::size_t id)
    {
        typename ComponentType::DisconnectWriterAction()(thisId.get(), id);
    }

    std::size_t getPeriod() const
    {
        return period;
    }

    std::size_t numUpdateGroups() const
    {
        return typename ComponentType::NumUpdateGroupsAction()(thisId.get());
    }

private:
    hpx::future<hpx::naming::id_type> thisId;
    std::size_t period;
    std::vector<hpx::future<void> > stepFinishedFutures;
    MutexType mtx;

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
