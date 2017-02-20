#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>
#include <libgeodecomp/geometry/adjacency.h>

#ifdef LIBGEODECOMP_WITH_CPP14
#ifdef LIBGEODECOMP_WITH_SCOTCH

#include <ptscotch.h>

#include <chrono>

namespace LibGeoDecomp {

/**
 * This class is like DistributedPTScotchUnstructuredPartition, but
 * relies on PT-SCOTCH's serial decomposition. This limits
 * scalability, but decouples the code from MPI.
 */
template<int DIM>
class PTScotchUnstructuredPartition : public Partition<DIM>
{
public:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;
    using typename Partition<DIM>::AdjacencyPtr;

    PTScotchUnstructuredPartition(
            const Coord<DIM>& origin,
            const Coord<DIM>& dimensions,
            const long offset,
            const std::vector<std::size_t>& weights,
            const AdjacencyPtr& adjacency) :
        Partition<DIM>(offset, weights),
        adjacency(adjacency),
        origin(origin),
        dimensions(dimensions),
        numCells(dimensions.prod())
    {
        buildRegions();
    }

    Region<DIM> getRegion(const std::size_t node) const override
    {
        return regions.at(node);
    }

private:
    SharedPtr<Adjacency>::Type adjacency;
    Coord<DIM> origin;
    Coord<DIM> dimensions;
    SCOTCH_Num numCells;
    std::vector<Region<DIM> > regions;

    void buildRegions()
    {
        std::vector<SCOTCH_Num> indices(numCells);
        initIndices(indices);

        regions.resize(weights.size());
        createRegions(indices);
    }

    void initIndices(std::vector<SCOTCH_Num>& indices)
    {
        // create 2D grid
        SCOTCH_Graph graph;
        int error = SCOTCH_graphInit(&graph);

        SCOTCH_Num numEdges = this->adjacency->size();

        std::vector<SCOTCH_Num> verttabGra;
        std::vector<SCOTCH_Num> edgetabGra;

        verttabGra.reserve(numCells + 1);
        edgetabGra.reserve(numEdges);

        int currentEdge = 0;
        for (int i = 0; i < numCells; ++i) {
            verttabGra.push_back(currentEdge);

            std::size_t before = edgetabGra.size();
            this->adjacency->getNeighbors(i, &edgetabGra);

            currentEdge += edgetabGra.size() - before;
        }

        verttabGra.push_back(currentEdge);

        error = SCOTCH_graphBuild(
                &graph,
                0,
                numCells,
                &verttabGra[0],
                nullptr,
                nullptr,
                nullptr,
                numEdges,
                &edgetabGra[0],
                nullptr);
        if (error) {
            LOG(ERROR, "SCOTCH_graphBuild error: " << error);
        }

        error = SCOTCH_graphCheck(&graph);
        if (error) {
            LOG(ERROR, "SCOTCH_graphCheck error: " << error);
        }

        SCOTCH_Strat *straptr = SCOTCH_stratAlloc();
        error = SCOTCH_stratInit(straptr);
        if (error) {
            LOG(ERROR, "SCOTCH_stratInit error: " << error);
        }

        error = SCOTCH_graphPart(&graph, weights.size(), straptr,& indices[0]);
        if (error) {
            LOG(ERROR, "SCOTCH_graphMap error: " << error);
        }

        SCOTCH_graphExit(&graph);
        SCOTCH_stratExit(straptr);
    }

    void createRegions(const std::vector<SCOTCH_Num>& indices)
    {
        for (int i = 0; i < numCells; ++i) {
            regions[indices[i]] << Coord<1>(i);
        }
    }

};

}

#endif
#endif

#endif

