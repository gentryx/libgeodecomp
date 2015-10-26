#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>
#include <libgeodecomp/geometry/adjacency.h>

#ifdef LIBGEODECOMP_WITH_SCOTCH
#ifdef LIBGEODECOMP_WITH_MPI

#include <mpi.h>
#include <ptscotch.h>

namespace LibGeoDecomp
{

template<int DIM>
class PTScotchUnstructuredPartition : public Partition<DIM>
{
    public:
        using Partition<DIM>::startOffsets;
        using Partition<DIM>::weights;

        explicit PTScotchUnstructuredPartition(
                const Coord<DIM> &origin = Coord<DIM>(),
                const Coord<DIM> &dimensions = Coord<DIM>(),
                const long &offset = 0,
                const std::vector<std::size_t> &weights = std::vector<std::size_t>(2)) :
            Partition<DIM>(offset, weights),
            origin(origin),
            dimensions(dimensions),
            numCells(dimensions.prod())
        {
        }

        Region<DIM> getRegion(const std::size_t node) const override
        {
            return regions.at(node);
        }

        virtual void setAdjacency(const Adjacency &in) override
        {
            std::cout << "set adjacency" << std::endl;

            adjacency = in;
            buildRegions();
        }

        virtual void setAdjacency(Adjacency &&in) override
        {
            std::cout << "move adjacency" << std::endl;

            adjacency = std::move(in);
            buildRegions();
        }

    private:
        Coord<DIM> origin;
        Coord<DIM> dimensions;
        SCOTCH_Num numCells;
        std::vector<Region<DIM> > regions;
        Adjacency adjacency;

        void buildRegions()
        {
            std::vector<SCOTCH_Num> indices(numCells);
            initIndices(indices);
            regions.resize(weights.size());
            createRegions(indices);
        }

        void initIndices(std::vector<SCOTCH_Num> &indices)
        {
            // create 2D grid
            SCOTCH_Arch arch;
            SCOTCH_archInit(&arch);
            SCOTCH_Num *velotabArch;
            SCOTCH_Num const vertnbrArch = weights.size();
            velotabArch = new SCOTCH_Num[weights.size()];
            for (int i = 0; i < vertnbrArch; ++i)
            {
                velotabArch[i] = weights[i];
            }
            int error = SCOTCH_archCmpltw(&arch, vertnbrArch, velotabArch);
            if (error) std::cout << "SCOTCH_archCmpltw error: " << error << std::endl;

            SCOTCH_Graph graph;
            error = SCOTCH_graphInit(&graph);

            SCOTCH_Num numEdges = 0;

            for (auto &p : adjacency)
            {
                numEdges += p.second.size();
            }

            SCOTCH_Num *verttabGra;
            SCOTCH_Num *edgetabGra;
            verttabGra = new SCOTCH_Num[numCells + 1];
            edgetabGra = new SCOTCH_Num[numEdges];

            int currentEdge = 0;
            for (int i = 0; i < numCells; ++i)
            {
                verttabGra[i] = currentEdge;

                for (int other : adjacency[i])
                {
                    edgetabGra[currentEdge++] = other;
                }
            }

            verttabGra[numCells] = currentEdge;

            error = SCOTCH_graphBuild(&graph,
                    0,
                    numCells,
                    verttabGra,
                    nullptr,
                    nullptr,
                    nullptr,
                    numEdges,
                    edgetabGra,
                    nullptr);
            if (error) std::cout << "SCOTCH_graphBuild error: " << error << std::endl;

            error = SCOTCH_graphCheck(&graph);
            if (error) std::cout << "SCOTCH_graphCheck error: " << error << std::endl;

            SCOTCH_Strat *straptr = SCOTCH_stratAlloc();
            error = SCOTCH_stratInit(straptr);
            if (error) std::cout << "SCOTCH_stratInit error: " << error << std::endl;

            error = SCOTCH_graphMap(&graph, &arch, straptr, &indices[0]);
            if (error) std::cout << "SCOTCH_graphMap error: " << error << std::endl;

            SCOTCH_archExit(&arch);
            SCOTCH_graphExit(&graph);
            SCOTCH_stratExit(straptr);
            delete[] velotabArch;
            delete[] verttabGra;
            delete[] edgetabGra;
        }

        void createRegions(const std::vector<SCOTCH_Num> &indices)
        {
            for (int i = 0; i < numCells; ++i)
            {
                regions[indices[i]] << Coord<1>(i);
            }

            if (MPILayer().rank() == 0)
            {
                std::cout << "regions: " << std::endl;
                for (auto &region : regions)
                {
                    region.prettyPrint1D(std::cout, Coord<2>(1000, 1000));
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
            }

        }

    protected:
    private:



};

}

#endif // LIBGEODECOMP_WITH_MPI
#endif // LIBGEODECOMP_WITH_SCOTCH

#endif // LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHUNSTRUCTUREDPARTITION_H

