
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_IO_SERVER_HPXWRITERSINK_H
#define LIBGEODECOMP_IO_SERVER_HPXWRITERSINK_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/local/spinlock.hpp>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

template <typename CELL_TYPE>
class IdentityConverter
{
public:
    typedef CELL_TYPE CellType;
    typedef typename CELL_TYPE::Topology Topology;

    const CellType& operator()(
        const CellType& cell,
        const Coord<Topology::DIM>& globalDimensions,
        unsigned step,
        std::size_t rank)
    {
        return cell;
    }
};

template<typename CELL_TYPE, typename CONVERTER = IdentityConverter<CELL_TYPE> >
class HpxWriterSinkServer
  : public hpx::components::managed_component_base<
        HpxWriterSinkServer<CELL_TYPE, CONVERTER>
    >
{

    class RegionInfo;

public:

    typedef typename CONVERTER::CellType CellType;
    static const int DIM = CellType::Topology::DIM;
    typedef typename CellType::Topology Topology;
    typedef Grid<CellType, Topology> GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;
    typedef SuperVector<CellType> BufferType;
    typedef SuperMap<unsigned, SuperVector<RegionInfo> > RegionInfoMapType;
    typedef SuperMap<unsigned, std::size_t> StepCountMapType;
    typedef SuperMap<unsigned, GridType> GridMapType;
    typedef
        SuperMap<std::size_t, boost::shared_ptr<ParallelWriter<CellType> > >
        ParallelWritersMap;
    typedef
        SuperMap<std::size_t, boost::shared_ptr<Writer<CellType> > >
        SerialWritersMap;

    typedef hpx::lcos::local::spinlock MutexType;

    HpxWriterSinkServer()
    {}

    HpxWriterSinkServer(
        std::size_t numUpdateGroups) :
        numUpdateGroups(numUpdateGroups),
        nextId(0)
    {
    }

    HpxWriterSinkServer(
        boost::shared_ptr<ParallelWriter<CellType> > parallelWriter,
        std::size_t numUpdateGroups) :
        numUpdateGroups(numUpdateGroups),
        nextId(0)
    {
        connectParallelWriter(parallelWriter);
    }

    HpxWriterSinkServer(
        boost::shared_ptr<Writer<CellType> > serialWriter,
        std::size_t numUpdateGroups) :
        numUpdateGroups(numUpdateGroups),
        nextId(0)
    {
        connectSerialWriter(serialWriter);
    }

    void stepFinished(
        boost::shared_ptr<BufferType> buffer,
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

        GridIterator kt;
        {
            MutexType::scoped_lock l(gridMapMtx);
            kt = gridMap.find(step);
            if(kt == gridMap.end())
            {
                kt = gridMap.insert(
                    kt,
                    std::make_pair(
                        step,
                        GridType(globalDimensions)));
            }
        }

        HiParSimulator::GridVecConv::vectorToGrid(
            *buffer,
            &kt->second,
            validRegion);

        RegionMapIterator it;
        {
            MutexType::scoped_lock l(regionMapMtx);
            it = regionInfoMap.find(step);
            if(it == regionInfoMap.end()) {
                it = regionInfoMap.insert(
                        it,
                        std::make_pair(step, SuperVector<RegionInfo>())
                    );
            }

            it->second.push_back(
                RegionInfo(validRegion, globalDimensions, event, rank, lastCall)
            );
        }

        if(lastCall) {
            MutexType::scoped_lock l0(stepCountMapMtx);
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
                    hpx::util::unlock_the_lock<MutexType::scoped_lock> ull(l0);
                    notifyWriters(kt->second, step, event);
                }
                {
                    MutexType::scoped_lock l1(regionMapMtx);
                    MutexType::scoped_lock l2(gridMapMtx);
                    regionInfoMap.erase(it);
                    stepCountMap.erase(jt);
                    gridMap.erase(kt);
                }
            }
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, stepFinished, StepFinishedAction);

    std::size_t connectParallelWriter(
        boost::shared_ptr<ParallelWriter<CellType> > parallelWriter)
    {
        MutexType::scoped_lock l(parallelWriterMapMtx);
        std::size_t id = getNextId();
        parallelWriters.insert(std::make_pair(id, parallelWriter));
        return id;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, connectParallelWriter, ConnectParallelWriterAction);

    std::size_t connectSerialWriter(
        boost::shared_ptr<Writer<CellType> > serialWriter)
    {
        MutexType::scoped_lock l(serialWriterMapMtx);
        std::size_t id = getNextId();
        serialWriters.insert(std::make_pair(id, serialWriter));
        return id;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, connectSerialWriter, ConnectSerialWriterAction);

    void disconnectWriter(std::size_t id)
    {
        {
            MutexType::scoped_lock l(parallelWriterMapMtx);
            typename ParallelWritersMap::iterator it = parallelWriters.find(id);
            if(it != parallelWriters.end()) {
                parallelWriters.erase(it);
                MutexType::scoped_lock l(freeIdsMtx);
                freeIds.push_back(id);
                return;
            }
        }
        {
            MutexType::scoped_lock l(serialWriterMapMtx);
            typename SerialWritersMap::iterator it = serialWriters.find(id);
            if(it != serialWriters.end()) {
                serialWriters.erase(it);
                MutexType::scoped_lock l(freeIdsMtx);
                freeIds.push_back(id);
                return;
            }
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, disconnectWriter, DisconnectWriterAction);

    std::size_t getNumUpdateGroups()
    {
        return numUpdateGroups;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxWriterSinkServer, getNumUpdateGroups, NumUpdateGroupsAction);

private:
    MutexType gridMapMtx;
    GridMapType gridMap;
    MutexType parallelWriterMapMtx;
    ParallelWritersMap parallelWriters;
    MutexType serialWriterMapMtx;
    SerialWritersMap serialWriters;
    std::size_t numUpdateGroups;

    MutexType stepCountMapMtx;
    StepCountMapType stepCountMap;
    MutexType regionMapMtx;
    RegionInfoMapType regionInfoMap;

    MutexType freeIdsMtx;
    std::size_t nextId;
    SuperVector<std::size_t> freeIds;

    std::size_t getNextId()
    {
        MutexType::scoped_lock l(freeIdsMtx);
        std::size_t id = 0;
        if(!freeIds.empty()) {
            id = freeIds.back();
            freeIds.pop_back();
        }
        else {
            id = nextId++;
        }
        return id;
    }

    void notifyWriters(GridType const & grid, unsigned step, WriterEvent event)
    {
        {
            MutexType::scoped_lock l(parallelWriterMapMtx);
            BOOST_FOREACH(typename ParallelWritersMap::value_type& writer,
                          parallelWriters) {
                MutexType::scoped_lock l(regionMapMtx);
                typedef typename RegionInfoMapType::iterator RegionInfoIterator;

                RegionInfoIterator it = regionInfoMap.find(step);
                BOOST_ASSERT(it != regionInfoMap.end());
                BOOST_FOREACH(RegionInfo const & regionInfo, it->second) {
                    writer.second->stepFinished(
                        grid,
                        regionInfo.validRegion,
                        regionInfo.globalDimensions,
                        step,
                        regionInfo.event, regionInfo.rank,
                        regionInfo.lastCall
                    );
                }
            }
        }
        {
            MutexType::scoped_lock l(serialWriterMapMtx);
            BOOST_FOREACH(typename SerialWritersMap::value_type& writer,
                          serialWriters) {
                writer.second->stepFinished(grid, step, event);
            }
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
