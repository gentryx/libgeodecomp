
#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHDISTRIBUTEDPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHDISTRIBUTEDPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>
#include <libgeodecomp/geometry/adjacency.h>

#ifdef LIBGEODECOMP_WITH_CPP14
#ifdef LIBGEODECOMP_WITH_SCOTCH
#ifdef LIBGEODECOMP_WITH_MPI

#include <mpi.h>
#include <ptscotch.h>
#include <chrono>

#ifdef SCOTCH_PTHREAD
#error can only use ptscotch if compiled without SCOTCH_PTHREAD
#endif // SCOTCH_PTHREAD


namespace LibGeoDecomp {

template<int DIM>
class PTScotchDistributedPartition : public Partition<DIM>
{
public:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    PTScotchDistributedPartition(
            const Coord<DIM> &origin,
            const Coord<DIM> &dimensions,
            const long &offset,
            const std::vector<std::size_t> &weights,
            const Adjacency &adjacency) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions),
        numPartitions(weights.size())
    {
        this->adjacency = adjacency;

        getStartEnd(MPILayer().rank(), start, end);
        localCells = end - start;
        buildRegions();
    }

    PTScotchDistributedPartition(
            const Coord<DIM> &origin,
            const Coord<DIM> &dimensions,
            const long &offset,
            const std::vector<std::size_t> &weights,
            Adjacency &&adjacency) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions),
        numPartitions(weights.size())
    {
        this->adjacency = std::move(adjacency);

        getStartEnd(MPILayer().rank(), start, end);
        localCells = end - start;
        buildRegions();
    }

    void getStartEnd(int rank, size_t &start, size_t &end)
    {
        size_t cells = dimensions.prod();
        start = (float)rank / numPartitions * cells;
        end = (float)(rank + 1) / numPartitions * cells;
    }

    Region<DIM> getRegion(const std::size_t node) const override
    {
        return regions.at(node);
    }

private:
    Coord<DIM> origin;
    Coord<DIM> dimensions;
    SCOTCH_Num localCells = 0;
    size_t start = 0, end = 0;
    size_t numPartitions = 0;
    std::vector<Region<DIM> > regions;

    void buildRegions()
    {
        std::vector<SCOTCH_Num> indices(localCells);

        {
            auto start = std::chrono::system_clock::now();
            initIndices(indices);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::system_clock::now() - start);
            std::cout << "partitioning duration: " << duration.count() << std::endl;
        }

        regions.resize(numPartitions);

        {
            auto start = std::chrono::system_clock::now();
            createRegions(indices);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::system_clock::now() - start);
            std::cout << "region creation duration: " << duration.count() << std::endl;

        }
    }

    void initIndices(std::vector<SCOTCH_Num> &indices)
    {
        std::cout << "partitioning ..." << std::endl;

        // create 2D grid
        SCOTCH_Dgraph graph;
        int error = SCOTCH_dgraphInit(&graph, MPILayer().communicator());

        SCOTCH_Num numEdges = 0;

        for (auto &p : this->adjacency)
        {
            numEdges += p.second.size();
        }

        SCOTCH_Num *verttabGra = new SCOTCH_Num[localCells + 1];
        SCOTCH_Num *edgetabGra = new SCOTCH_Num[numEdges];

        int currentEdge = 0;
        for (int i = 0; i < localCells; ++i)
        {
            verttabGra[i] = currentEdge;

            for (int other : this->adjacency[start + i])
            {
                edgetabGra[currentEdge++] = other;
            }
        }

        numEdges = currentEdge;

        verttabGra[localCells] = currentEdge;

        int ierr = 0;
        error = SCOTCH_dgraphBuild(&graph,
                0,              // c++ starts counting at 0
                localCells,     // number of local vertices
                localCells,     // number of maximum local vertices to be created
                verttabGra,     // vertex data
                nullptr,        // these are some ghost zone
                nullptr,        // and other hints that are optional, so -> nullptr
                nullptr,        //
                numEdges,       // number of local edges
                numEdges,       // number of maximum local edges to be created
                edgetabGra,     // edge data
                nullptr,        // more optional data/hints
                nullptr);       //
        if (error) std::cout << "SCOTCH_graphBuild error: " << error << ", ierr: " << ierr << std::endl;

        error = SCOTCH_dgraphCheck(&graph);
        if (error) std::cout << "SCOTCH_graphCheck error: " << error << std::endl;

        SCOTCH_Strat *straptr = SCOTCH_stratAlloc();
        error = SCOTCH_stratInit(straptr);
        if (error) std::cout << "SCOTCH_stratInit error: " << error << std::endl;

        error = SCOTCH_dgraphPart(&graph, numPartitions, straptr, &indices[0]);
        if (error) std::cout << "SCOTCH_graphMap error: " << error << std::endl;

        SCOTCH_dgraphExit(&graph);
        SCOTCH_stratExit(straptr);
        delete[] verttabGra;
        delete[] edgetabGra;
    }

    void createRegions(const std::vector<SCOTCH_Num> &indices)
    {
        // regions won't be complete after partitioning, fragments of different regions
        // will end up on different nodes and they have to be synchronized.
        // build partial regions
        std::vector<Region<1>> partials(numPartitions);
        for (int i = 0; i < indices.size(); ++i)
        {
            partials.at(indices[i]) << Coord<1>(start + i);
        }
        regions = partials;

        // send partial regions to all other nodes so they can build the complete regions
        size_t numIndices = indices.size();
        for (size_t j = 0; j < numPartitions; ++j)
        {
            if (j != MPILayer().rank()) // dont send own regions to self
            {
                // send all partial regions
                for (size_t k = 0; k < numPartitions; ++k)
                {
                    MPILayer().sendRegion(partials.at(k), j);
                }
            }
            else
            {
                // regions will be received from all ranks except self
                for (int i = 0; i < numPartitions; ++i)
                {
                    if (i == j) continue; // we won't receive regions from ourself

                    // receive parts & build up own regions
                    for (size_t k = 0; k < numPartitions; ++k)
                    {
                        Region<1> received;
                        MPILayer().recvRegion(&received, i);

                        // add the region to ourself
                        regions.at(k) += received;
                    }
                }
            }
        }

        // pretty print regions. those should all be the same in the end
#if 1
#ifdef GRID_SIZE
        MPILayer().barrier();

        for (int i = 0; i < MPILayer().size(); ++i)
        {
            if (MPILayer().rank() == i)
            {
                std::cout << "regions of rank " << i << ": " << std::endl;
                std::stringstream ss;
                Region<1>::prettyPrint1D(ss, regions, Coord<2>(GRID_SIZE, GRID_SIZE), Coord<2>(16, 10));
                std::cout << ss.str() << std::endl;
            }

            MPILayer().barrier();
        }
#endif // GRID_SIZE
#endif // 0

    }

};

}

#endif
#endif
#endif

#endif

