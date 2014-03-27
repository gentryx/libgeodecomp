#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_PTSCOTCHPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>

#ifdef LIBGEODECOMP_WITH_SCOTCH
#ifdef LIBGEODECOMP_WITH_MPI

#include <mpi.h>
#include <ptscotch.h>

namespace LibGeoDecomp {

template<int DIM>
class PTScotchPartition : public Partition<DIM>
{
public:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    inline PTScotchPartition(
        const Coord<DIM>& origin = Coord<DIM>(),
        const Coord<DIM>& dimensions = Coord<DIM>(),
        const long& offset = 0,
        const std::vector<std::size_t>& weights = std::vector<std::size_t>(2)) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions)
        {
            if(DIM == 3){
                cellNbr = dimensions[0] *
                    dimensions[1] *
                    dimensions[2];
            } else {
                cellNbr = dimensions[0] *
                    dimensions[1];
            }
            indices = new SCOTCH_Num[cellNbr];
            initIndices();
            regions = new Region<DIM>[weights.size()];
            createRegions();
        }

    Region<DIM> getRegion(const std::size_t node) const
    {
        return regions[node];
    }

private:
    Coord<DIM> origin;
    Coord<DIM> dimensions;
    SCOTCH_Num *indices;
    SCOTCH_Num cellNbr;
    Region<DIM> * regions;

    void initIndices(){
        SCOTCH_Arch arch;
        SCOTCH_archInit(&arch);
        SCOTCH_Num * velotabArch;
        SCOTCH_Num const vertnbrArch = weights.size();
        velotabArch = new SCOTCH_Num[weights.size()];
        for(int i = 0; i < vertnbrArch; ++i){
            velotabArch[i] = weights[i];
        }
        SCOTCH_archCmpltw(&arch, vertnbrArch, velotabArch);

        SCOTCH_Dgraph grafdat;
        SCOTCH_dgraphInit(&grafdat, MPI_COMM_WORLD);

        SCOTCH_Num edgenbrGra = 2 * (dimensions[0] *
                                     (dimensions[1] - 1) +
                                     (dimensions[0] - 1) * dimensions[1]);
        if(DIM == 3){
            edgenbrGra = edgenbrGra
                * dimensions[2]
                + 2 * (dimensions[0] *
                       dimensions[1] * (dimensions[2] - 1));
        }

        SCOTCH_Num *verttabGra;
        SCOTCH_Num *edgetabGra;
        verttabGra = new SCOTCH_Num[cellNbr + 1];
        edgetabGra = new SCOTCH_Num[edgenbrGra];

        int pointer = 0;
        int xyArea = dimensions[0] * dimensions[1];
        for(int i = 0; i < cellNbr; ++i){
            verttabGra[i] = pointer;
            if(i % dimensions[0] != 0){
                edgetabGra[pointer] = i - 1;
                pointer++;
            }
            if(i % dimensions[0] != (dimensions[0] - 1)){
                edgetabGra[pointer] = i + 1;
                pointer++;
            }
            if(!((i % xyArea) < dimensions[0])){
                edgetabGra[pointer] = i - dimensions[0];
                pointer++;
            }
            if(!((i % xyArea) >= dimensions[0] * (dimensions[1] - 1))){
                edgetabGra[pointer] = i + dimensions[0];
                pointer++;
            }
            if(DIM == 3 && i >= (dimensions[0] * dimensions[1])){
                edgetabGra[pointer] = i - (dimensions[0] * dimensions[1]);
                pointer++;
            }
            if(DIM == 3 && i < (dimensions[0] *
                                dimensions[1]) * (dimensions[2] - 1)){
                edgetabGra[pointer] = i + (dimensions[0] * dimensions[1]);
                pointer++;
            }
        }
        verttabGra[cellNbr] = pointer;

        SCOTCH_dgraphBuild(&grafdat,
                           0,
                           cellNbr,
                           cellNbr,
                           verttabGra,
                           verttabGra +1,
                           NULL,
                           NULL,
                           edgenbrGra,
                           edgenbrGra,
                           edgetabGra,
                           NULL,
                           NULL);


        SCOTCH_Strat *straptr = SCOTCH_stratAlloc();;
        SCOTCH_stratInit(straptr);

        SCOTCH_dgraphMap (&grafdat, &arch, straptr, indices);

        SCOTCH_archExit (&arch);
        SCOTCH_dgraphExit (&grafdat);
        SCOTCH_stratExit (straptr);
        free (velotabArch);
        free (verttabGra);
        free (edgetabGra);

    }

    void createRegions(){
        int rank = indices[0];
        int length = 0;
        int start = 0;
        for(int i = 1; i < cellNbr + 1; ++i){
            if((rank == indices[i]) &&
               (i < cellNbr) && (i % dimensions[0] != 0)){
                length++;
            } else {
                length++;
                Coord<DIM> startCoord;
                Coord<DIM> lengthCoord;
                lengthCoord[0] = length;
                lengthCoord[1] = 1;
                if(DIM == 3){
                    startCoord[0] = origin[0] +
                        (start % (dimensions[0] * dimensions[1])) %
                        dimensions[0];
                    startCoord[1] = origin[1] +
                        (start % (dimensions[0] * dimensions[1])) /
                        dimensions[0];
                    startCoord[2] = origin[2] + start /
                        (dimensions[0] *
                         dimensions[1]);
                    lengthCoord[2] = 1;
                } else {
                    startCoord[0] = origin[0] + start % dimensions[0];
                    startCoord[1] = origin[1] + start / dimensions[0];
                }


                regions[rank] <<
                    CoordBox<DIM>(startCoord, lengthCoord);
                rank = indices[i];
                start = i;
                length = 0;
            }
        }
    }
 };
}

#endif

#endif

#endif
