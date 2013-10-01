
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
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef typename DistributedSimulator<CellType>::GridType GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;
    typedef std::vector<CellType> BufferType;
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

        while(thisId == hpx::naming::invalid_id) {
            hpx::agas::resolve_name_sync(name, thisId);
            if(retry > 10) {
                throw std::logic_error("Can't find the Writer Sink name");
            }
            hpx::this_thread::suspend();
            ++retry;
        }
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
                numUpdateGroups).move();
        hpx::agas::register_name(name, thisId);
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
                numUpdateGroups).move();
        if(name != "") {
            hpx::agas::register_name(name, thisId);
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
                numUpdateGroups).move();
        if(name != "") {
            hpx::agas::register_name(name, thisId);
        }
    }

    HpxWriterSink(const HpxWriterSink& sink) :
        thisId(sink.thisId),
        period(sink.period)
    {}

    void stepFinished(
        const typename DistributedSimulator<CELL_TYPE>::GridType& grid,
        const Region<Topology::DIM>& validRegion,
        const Coord<Topology::DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        boost::shared_ptr<BufferType> buffer(new BufferType(validRegion.size()));

        CellType *dest = &(*buffer)[0];

        for (typename Region<Topology::DIM>::Iterator i = validRegion.begin();
             i != validRegion.end(); ++i) {
            *dest = CONVERTER()(grid.get(*i), globalDimensions, step, rank);
            ++dest;
        }

        hpx::future<void> stepFinishedFuture
            = hpx::async<typename ComponentType::StepFinishedAction>(
                thisId,
                buffer,
                validRegion,
                globalDimensions,
                step,
                event,
                rank,
                lastCall
            );

        if(stepFinishedFutures.size() > 10) {
            std::vector<hpx::future<void> > res(hpx::wait_any(stepFinishedFutures));
            BOOST_FOREACH(hpx::future<void>& f, stepFinishedFutures)
            {
                if(f.is_ready()) {
                    f = stepFinishedFuture;
                }
            }
        }
        else {
            stepFinishedFutures.push_back(stepFinishedFuture);
        }
    }

    hpx::future<std::size_t> connectWriter(ParallelWriter<CellType> *parallelWriter)
    {
        boost::shared_ptr<ParallelWriter<CellType> > writer(parallelWriter);
        return
            hpx::async<typename ComponentType::ConnectParallelWriterAction>(
                thisId,
                writer);
    }

    hpx::future<std::size_t> connectWriter(Writer<CellType> *serialWriter)
    {
        boost::shared_ptr<Writer<CellType> > writer(serialWriter);
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

    std::size_t numUpdateGroups() const
    {
        return typename ComponentType::NumUpdateGroupsAction()(thisId);
    }

private:
    hpx::naming::id_type thisId;
    std::size_t period;
    std::vector<hpx::future<void> > stepFinishedFutures;

    template<typename ARCHIVE>
    void load(ARCHIVE& ar, unsigned)
    {
        ar & thisId;
        ar & period;
    }

    template<typename ARCHIVE>
    void save(ARCHIVE& ar, unsigned) const
    {
        ar & thisId;
        ar & period;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

}

#endif
#endif
