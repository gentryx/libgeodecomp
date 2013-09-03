#include <libgeodecomp.h>

using namespace LibGeoDecomp;

class CactusCell
{
public:
    typedef Stencils::Moore<3, 1> Stencil;

    class API :
        public CellAPITraits::Fixed,
        public CellAPITraitsFixme::HasTorusTopology<3>,
        public CellAPITraitsFixme::HasUpdateLineX
    {};

    // roughly imitating what's required by WaveToyC_Evolution in
    // "Cactus/arrangements/CactusWave/WaveToyC/src/WaveToy.c".
    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineX(ACCESSOR1 hoodOld, int *indexOld, int indexEnd, ACCESSOR2 hoodNew, int *indexNew)
    {
        std::cout << "CactusCell::updateLineX(" << indexEnd << ")\n";
//         // artificial code:
// // fixme #define GET_COMP(X, Y, Z, COMP) Double(&hoodOld[FixedCoord<X, Y, Z>()].COMP())
// // fixme: use dim_x, dim_y... to overload cactus vars
// // fixme: rotate phi_p, phi_p_p
// #define CCTK_GFINDEX3D(UNUSED, I, J, K) ((I) + (J) * ACCESSOR1::DIM_X1 + (K) * ACCESSOR1::DIM_X1 + ACCESSOR1::DIM_Y1)
// #define phi     &hoodNew[FixedCoord<0, 0, 0>()].var_phi
// #define phi_p   &hoodOld[FixedCoord<0, 0, 0>()].var_phi
// #define phi_p_p &hoodOld[FixedCoord<0, 0, 0>()].var_phi_p

//         int kstart = 0;
//         int kend = 1;
//         int jstart = 0;
//         int jend = 1;
//         int istart = 0;
//         int iend = indexEnd - *indexOld;
//         int i;
//         int j;
//         int k;

//         // original code:
//         double dx2i = 0.2;
//         double dy2i = 0.2;
//         double dz2i = 0.2;
//         double dt2 = 0.1;
//         double factor = 2*(1 - (dt2)*(dx2i + dy2i + dz2i));

//         for (k=kstart; k<kend; k++) {
//             for (j=jstart; j<jend; j++) {
//                 for (i=istart; i<iend; i++) {
//                     int vindex = CCTK_GFINDEX3D(cctkGH,i,j,k);

//                     phi[vindex] = factor*
//                      phi_p[vindex] - phi_p_p[vindex]
//                      + (dt2) *
//                         ( ( phi_p[CCTK_GFINDEX3D(cctkGH,i+1,j  ,k  )] +
//                             phi_p[CCTK_GFINDEX3D(cctkGH,i-1,j  ,k  )] )*dx2i +
//                           ( phi_p[CCTK_GFINDEX3D(cctkGH,i  ,j+1,k  )] +
//                             phi_p[CCTK_GFINDEX3D(cctkGH,i  ,j-1,k  )] )*dy2i +
//                           ( phi_p[CCTK_GFINDEX3D(cctkGH,i  ,j  ,k+1)] +
//                             phi_p[CCTK_GFINDEX3D(cctkGH,i  ,j,  k-1)] )*dz2i);
//                 }
//             }
//         }
    }

    double var_phi;
    double var_phi_p;
};

int main(int argc, char **argv)
{
    std::cout << "Kurt's lair\n";
}
