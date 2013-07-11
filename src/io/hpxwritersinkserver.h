
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_SERVER_HPXWRITERSINK_H
#define LIBGEODECOMP_IO_SERVER_HPXWRITERSINK_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

template <typename CELL_TYPE>
class HpxWriterSinkServer
  : public hpx::components::managed_component_base<
        HpxWriterSinkServer<CELL_TYPE>
    >
{

    class RegionInfo;

public:

    static const int DIM = CELL_TYPE::Topology::DIM;
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;
    typedef SuperVector<CELL_TYPE> BufferType;
    typedef SuperMap<unsigned, SuperVector<RegionInfo> > RegionInfoMapType;
    typedef SuperMap<unsigned, std::size_t> StepCountMapType;
    typedef SuperMap<unsigned, GridType> GridMapType;

    typedef hpx::lcos::local::spinlock MutexType;

    HpxWriterSinkServer()
    {}

    HpxWriterSinkServer(
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > parallelWriter,
        std::size_t numUpdateGroups) :
        parallelWriter(parallelWriter),
        numUpdateGroups(numUpdateGroups)
    {}

    HpxWriterSinkServer(
        boost::shared_ptr<Writer<CELL_TYPE> > serialWriter,
        std::size_t numUpdateGroups) :
        serialWriter(serialWriter),
        numUpdateGroups(numUpdateGroups)
    {}

    void stepFinished(
        const BufferType& buffer,
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        typedef typename RegionInfoMapType::iterator RegionMapIterator;
        typedef typename StepCountMapType::iterator StepCountMapIterator;
        typedef typename GridMapType::iterator GridIterator;

        MutexType::scoped_lock l(mtx);
        GridIterator kt = gridMap.find(step);
        if(kt == gridMap.end())
        {
            kt = gridMap.insert(
                kt,
                std::make_pair(
                    step,
                    GridType(globalDimensions)));
        }

        HiParSimulator::GridVecConv::vectorToGrid(buffer, &kt->second, validRegion);

        RegionMapIterator it = regionInfoMap.find(step);
        if(it == regionInfoMap.end()) {
            it = regionInfoMap.insert(
                    it,
                    std::make_pair(step, SuperVector<RegionInfo>())
                );
        }

        it->second.push_back(
            RegionInfo(validRegion, globalDimensions, event, rank, lastCall)
        );

        if(lastCall) {
            StepCountMapIterator jt = stepCountMap.find(step);
            if(jt == stepCountMap.end())
            {
                jt = stepCountMap.insert(jt, std::make_pair(step, 1));
            }
            else
            {
                ++jt->second;
            }

            if(jt->second == numUpdateGroups)
            {
                {
                    hpx::util::unlock_the_lock<MutexType::scoped_lock> ull(l);
                    notifyWriters(kt->second, step, event);
                }
                regionInfoMap.erase(it);
                stepCountMap.erase(jt);
                gridMap.erase(kt);
            }
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, stepFinished, StepFinishedAction);

private:
    GridMapType gridMap;
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > parallelWriter;
    boost::shared_ptr<Writer<CELL_TYPE> > serialWriter;
    std::size_t numUpdateGroups;

    StepCountMapType stepCountMap;
    RegionInfoMapType regionInfoMap;

    MutexType mtx;

    void notifyWriters(const GridType& grid, unsigned step, WriterEvent event)
    {
        if(!parallelWriter) {
            MutexType::scoped_lock l(mtx);
            typedef typename RegionInfoMapType::iterator RegionInfoIterator;

            RegionInfoIterator it = regionInfoMap.find(step);
            BOOST_ASSERT(it != regionInfoMap.end());
            BOOST_FOREACH(const RegionInfo& regionInfo, it->second) {
                parallelWriter->stepFinished(
                    grid,
                    regionInfo.validRegion,
                    regionInfo.globalDimensions,
                    step,
                    regionInfo.event, regionInfo.rank,
                    regionInfo.lastCall
                );
            }
        }
        if(serialWriter) {
            serialWriter->stepFinished(grid, step, event);
        }
    }

    class RegionInfo
    {
    public:
        RegionInfo(
            const RegionType& validRegion,
            const CoordType& globalDimensions,
            WriterEvent event,
            std::size_t rank,
            bool lastCall) :
            validRegion(validRegion),
            globalDimensions(globalDimensions),
            event(event),
            rank(rank),
            lastCall(lastCall)
        {}

        RegionType validRegion;
        CoordType globalDimensions;
        WriterEvent event;
        std::size_t rank;
        bool lastCall;
    };

};

}

#endif
#endif
