#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>
#include <libgeodecomp/geometry/adjacency.h>

#ifdef LIBGEODECOMP_WITH_CPP14
#ifdef LIBGEODECOMP_WITH_SCOTCH
#ifdef LIBGEODECOMP_WITH_MPI

#include <mpi.h>
#include <ptscotch.h>

#include <chrono>

namespace LibGeoDecomp {

template<int DIM>
class PTScotchUnstructuredPartition : public Partition<DIM>
{
public:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    PTScotchUnstructuredPartition(
            const Coord<DIM>& origin,
            const Coord<DIM>& dimensions,
            const long offset,
            const std::vector<std::size_t>& weights,
            const Adjacency& adjacency) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions),
        numCells(dimensions.prod())
    {
        this->adjacency = adjacency;
        buildRegions();
    }

    PTScotchUnstructuredPartition(
            const Coord<DIM>& origin,
            const Coord<DIM>& dimensions,
            const long offset,
            const std::vector<std::size_t>& weights,
            Adjacency&& adjacency) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions),
        numCells(dimensions.prod())
    {
        this->adjacency = std::move(adjacency);
        buildRegions();
    }

    Region<DIM> getRegion(const std::size_t node) const override
    {
        return regions.at(node);
    }

private:
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

        SCOTCH_Num numEdges = 0;

        // ok, this is SUPER ugly
        for (auto& p : this->adjacency.getRegion()) {
            numEdges ++;
        }

        std::vector<int> neighbors;

        SCOTCH_Num *verttabGra;
        SCOTCH_Num *edgetabGra;
        verttabGra = new SCOTCH_Num[numCells + 1];
        edgetabGra = new SCOTCH_Num[numEdges];

        int currentEdge = 0;
        for (int i = 0; i < numCells; ++i) {
            verttabGra[i] = currentEdge;

            this->adjacency.getNeighbors(i,& neighbors);
            for (int other : neighbors) {
                edgetabGra[currentEdge++] = other;
            }
        }

        verttabGra[numCells] = currentEdge;

        error = SCOTCH_graphBuild(
                &graph,
                0,
                numCells,
                verttabGra,
                nullptr,
                nullptr,
                nullptr,
                numEdges,
                edgetabGra,
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
        delete[] verttabGra;
        delete[] edgetabGra;
    }

    void createRegions(const std::vector<SCOTCH_Num>& indices)
    {
        for (int i = 0; i < numCells; ++i) {
            regions[indices[i]] << Coord<1>(i);
        }

#if 0
        if (MPILayer().rank() == 0) {
            LOG(DBG, "regions: ");
            for (auto& region : regions) {
                stringstream ss;
                region.prettyPrint1D(ss, Coord<2>(2000, 2000), Coord<2>(8, 8));
                LOG(DBG, ss.str());
            }
        }
#endif

    }

};

}

#endif
#endif
#endif

#endif

