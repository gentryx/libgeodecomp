
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

    explicit PTScotchDistributedPartition(
            const Coord<DIM> &origin,
            const Coord<DIM> &dimensions,
            const long &offset,
            const std::vector<std::size_t> &weights,
            const Adjacency &adjacency) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions),
        adjacency(adjacency)
    {
        int cells = dimensions.prod();
        start = (float)MPILayer().rank() / MPILayer().size() * cells;
        end = (float)(MPILayer().rank() + 1) / MPILayer().size() * cells;

        localCells = end - start;
        std::cout << "local cells: " << localCells << std::endl;

        buildRegions();
    }

    Region<DIM> getRegion(const std::size_t node) const override
    {
        return regions.at(node);
    }

    virtual const Adjacency &getAdjacency() const override
    {
        return adjacency;
    }

private:
    Coord<DIM> origin;
    Coord<DIM> dimensions;
    SCOTCH_Num localCells;
    int start, end;
    std::vector<Region<DIM> > regions;
    Adjacency adjacency;


    void buildRegions()
    {
        std::vector<SCOTCH_Num> indices(localCells);
        initIndices(indices);
        regions.resize(weights.size());
        createRegions(indices);
    }

    void initIndices(std::vector<SCOTCH_Num> &indices)
    {
        // create 2D grid
        SCOTCH_Dgraph graph;
        int error = SCOTCH_dgraphInit(&graph, MPILayer().communicator());

        SCOTCH_Num numEdges = 0;

        for (auto &p : adjacency)
        {
            numEdges += p.second.size();
        }

        SCOTCH_Num *verttabGra = new SCOTCH_Num[localCells + 1];
        SCOTCH_Num *edgetabGra = new SCOTCH_Num[numEdges];

        int currentEdge = 0;
        for (int i = 0; i < localCells; ++i)
        {
            verttabGra[i] = currentEdge;

            for (int other : adjacency[start + i])
            {
                edgetabGra[currentEdge++] = other;
            }
        }

        numEdges = currentEdge;

        std::cout << "local cells: " << localCells << std::endl;
        std::cout << "local edges: " << numEdges << std::endl;

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

        error = SCOTCH_dgraphPart(&graph, weights.size(), straptr, &indices[0]);
        if (error) std::cout << "SCOTCH_graphMap error: " << error << std::endl;

        SCOTCH_dgraphExit(&graph);
        SCOTCH_stratExit(straptr);
        delete[] verttabGra;
        delete[] edgetabGra;
    }

    void createRegions(const std::vector<SCOTCH_Num> &indices)
    {
        std::cout << "building regions on " << MPILayer().rank() << std::endl;
        std::stringstream ss;

        std::vector<Region<1>> partialRegions(weights.size());

        ss << "rank " << MPILayer().rank() << " indices: ";
        for (int i = 0; i < localCells; ++i)
        {
            ss << "(" << indices[i] << "," << (start + i) << ") ";

            partialRegions.at(indices[i]) << Coord<1>(start + i);
        }

        std::cout << ss.str() << std::endl;

        // to all ranks
        for (size_t j = 0; j < weights.size(); ++j)
        {
            // send all regions
            for (unsigned int i = 0; i < partialRegions.size(); ++i)
            {
                MPILayer().sendRegion(partialRegions[i], j);
            }
        }

        for (size_t j = 0; j < weights.size(); ++j)
        {
            // receive all regions
            for (unsigned int i = 0; i < partialRegions.size(); ++i)
            {
                Region<1> add;
                MPILayer().recvRegion(&add, j);
                regions[i] += add;
            }
        }

        std::cout << regions.at(MPILayer().rank()) << std::endl;
        std::cout << std::flush;
        MPILayer().barrier();

#ifdef GRID_SIZE
        for (unsigned int i = 0; i < MPILayer().size(); ++i)
        {
            if (MPILayer().rank() == i)
            {
                std::stringstream ss;

                for (unsigned j = 0; j < regions.size(); ++j)
                {
                    ss << i << ": region #" << j << ": " << std::endl;
                    regions[j].prettyPrint1D(ss, Coord<2>(GRID_SIZE, GRID_SIZE));
                    ss << std::endl;
                }

                std::cout << ss.str() << std::endl;
            }

            MPILayer().barrier();
        }
#endif // GRID_SIZE

    }

};

}

#endif
#endif
#endif

#endif

