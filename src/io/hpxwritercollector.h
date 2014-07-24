#ifndef LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H
#define LIBGEODECOMP_IO_HPXWRITERCOLLECTOR_H

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename CONVERTER>
class HpxWriterCollector;

}

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/hpxwritersink.h>
#include <libgeodecomp/misc/clonable.h>

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
        TYPE ::SinkType::ComponentType::NumUpdateGroupsAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), NumUpdateGroupsAction)       \
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
        TYPE ::SinkType::ComponentType::NumUpdateGroupsAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, SinkType), NumUpdateGroupsAction)       \
    )                                                                           \
                                                                                \
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

template<typename CELL_TYPE, typename CONVERTER = IdentityConverter<CELL_TYPE> >
class HpxWriterCollector : public Clonable<ParallelWriter<CELL_TYPE>, HpxWriterCollector<CELL_TYPE, CONVERTER> >
{
public:
    friend class Serialization;
    friend class boost::serialization::access;

    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;

    static const int DIM = Topology::DIM;

    using ParallelWriter<CELL_TYPE>::period;
    using ParallelWriter<CELL_TYPE>::prefix;
    typedef typename ParallelWriter<CELL_TYPE>::GridType GridType;
    typedef typename ParallelWriter<CELL_TYPE>::RegionType RegionType;
    typedef typename ParallelWriter<CELL_TYPE>::CoordType CoordType;

    typedef HpxWriterSink<CELL_TYPE, CONVERTER> SinkType;

    explicit HpxWriterCollector(const SinkType& sink = SinkType()) :
        Clonable<ParallelWriter<CELL_TYPE>, HpxWriterCollector<CELL_TYPE, CONVERTER> >(
            "",
            sink.getPeriod()),
        sink(sink)
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
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        sink.stepFinished(grid, validRegion, globalDimensions, step, event, rank, lastCall);
    }

private:
    SinkType sink;
};

class Serialization;

}

#endif
#endif
