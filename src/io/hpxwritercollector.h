
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H
#define LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/hpxwritersink.h>

#define LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR_DECLARATION(TYPE, NAME)      \
    BOOST_CLASS_EXPORT_KEY2(TYPE, BOOST_PP_STRINGIZE(NAME));                    \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentType::StepFinishedAction,                     \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), StepFinishedAction)          \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentType::ConnectParallelWriterAction,            \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ConnectParallelWriterAction) \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentType::ConnectSerialWriterAction,              \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ConnectSerialWriterAction)   \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentType::DisconnectWriterAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), DisconnectWriterAction)      \
    )                                                                           \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentCreateActionType,                             \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), CreateAction)                \
    )                                                                           \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentWriterCreateActionType,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), WriterCreateAction)          \
    )                                                                           \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        TYPE ::SinkType::ComponentParallelWriterCreateActionType,               \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ParallelWriterCreateAction)  \
    )                                                                           \
/**/

#define LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR(TYPE, NAME)                  \
    BOOST_CLASS_EXPORT_IMPLEMENT(TYPE);                                         \
    typedef                                                                     \
        hpx::components::managed_component<                                     \
            TYPE ::SinkType::ComponentType                                      \
        >                                                                       \
        BOOST_PP_CAT(NAME, SinkComponentType);                                  \
                                                                                \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                     \
        BOOST_PP_CAT(NAME, SinkComponentType),                                  \
        BOOST_PP_CAT(NAME, SinkComponentType)                                   \
    );                                                                          \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentType::StepFinishedAction,                     \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), StepFinishedAction)          \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentType::ConnectParallelWriterAction,            \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ConnectParallelWriterAction) \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentType::ConnectSerialWriterAction,              \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ConnectSerialWriterAction)   \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentType::DisconnectWriterAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), DisconnectWriterAction)      \
    )                                                                           \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentCreateActionType,                             \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), CreateAction)                \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentWriterCreateActionType,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), WriterCreateAction)          \
    )                                                                           \
                                                                                \
    HPX_REGISTER_ACTION(                                                        \
        TYPE ::SinkType::ComponentParallelWriterCreateActionType,               \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), ParallelWriterCreateAction)  \
    )                                                                           \
/**/

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

    typedef HpxWriterSink<CELL_TYPE> SinkType;

    HpxWriterCollector(HpxWriterSink<CELL_TYPE> const & sink) :
        ParallelWriter<CELL_TYPE>("", sink.getPeriod()),
        sink(sink)
    {
    }

    ParallelWriter<CELL_TYPE> * clone()
    {
        return new HpxWriterCollector(sink);
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
    SinkType sink;

    HpxWriterCollector() {}

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
