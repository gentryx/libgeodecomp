#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/linepointerassembly.h>
#include <libgeodecomp/storage/linepointerupdatefunctor.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>
#include <libgeodecomp/storage/unstructuredupdatefunctor.h>

#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/macros.hpp>

#include <emmintrin.h>
#ifdef __AVX__
#include <immintrin.h>
#endif

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>

#include "mmio.h"

using namespace LibGeoDecomp;
using namespace LibFlatArray;

// VALUE_TYPE = double, MATRICES = 1
// FIXME: Indention is broken
#define SOA_CELL(C, SIGMA)                                              \
    class SPMVMSoACell ## _ ## C ## _ ## SIGMA                          \
    {                                                                   \
    public:                                                             \
        class API :                                                     \
            public APITraits::HasSoA,                                   \
            public APITraits::HasUpdateLineX,                           \
            public APITraits::HasUnstructuredTopology,                  \
            public APITraits::HasSellType<double>,                      \
            public APITraits::HasSellMatrices<1>,                       \
            public APITraits::HasSellC<C>,                              \
            public APITraits::HasSellSigma<SIGMA>                       \
            {                                                           \
            public:                                                     \
    LIBFLATARRAY_CUSTOM_SIZES(                                          \
        (16)(32)(64)(128)(256)(512)(1024)(2048)(4096)(8192)(16384)(32768) \
        (65536)(131072)(262144)(524288)(1048576)(2097152)(4194304),     \
        (1),                                                            \
        (1))                                                            \
        };                                                              \
                                                                        \
typedef short_vec<double, C> ShortVec;                                  \
                                                                        \
inline explicit SPMVMSoACell ## _ ## C ## _ ## SIGMA(double v = 8.0) :  \
    value(v), sum(0)                                                    \
{}                                                                      \
                                                                        \
template<typename HOOD_NEW, typename HOOD_OLD>                          \
static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */) \
{                                                                       \
    const auto& membersOld = hoodOld.accessor;                          \
    auto& membersNew = hoodNew.accessor;                                \
    for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {       \
        ShortVec tmp;                                                   \
        tmp.load_aligned(&membersNew.sum() + i * C);                    \
        for (const auto& j: hoodOld.weights(0)) {                       \
            ShortVec weights, values;                                   \
            weights.load_aligned(j.second);                             \
            values.gather(&membersOld.value(), j.first);                \
            tmp += values * weights;                                    \
        }                                                               \
        tmp.store_aligned(&membersNew.sum() + i * C);                   \
    }                                                                   \
}                                                                       \
                                                                        \
template<typename NEIGHBORHOOD>                                         \
void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)        \
{                                                                       \
    sum = 0.;                                                           \
    for (const auto& j: neighborhood.weights(0)) {                      \
        sum += neighborhood[j.first].value * j.second;                  \
    }                                                                   \
}                                                                       \
                                                                        \
double value;                                                           \
double sum;                                                             \
    };                                                                  \
                                                                        \
    LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell ## _ ## C ## _ ## SIGMA, ((double)(sum))((double)(value))) \


// create CELL classes
SOA_CELL(4 , 1)
SOA_CELL(8 , 1)
SOA_CELL(16, 1)
SOA_CELL(4 , 2)
SOA_CELL(8 , 2)
SOA_CELL(16, 2)
SOA_CELL(4 , 4)
SOA_CELL(8 , 4)
SOA_CELL(16, 4)
SOA_CELL(4 , 16)
SOA_CELL(8 , 16)
SOA_CELL(16, 16)
SOA_CELL(4 , 32)
SOA_CELL(8 , 32)
SOA_CELL(16, 32)
SOA_CELL(4 , 64)
SOA_CELL(8 , 64)
SOA_CELL(16, 64)
SOA_CELL(4 , 128)
SOA_CELL(8 , 128)
SOA_CELL(16, 128)
SOA_CELL(4 , 256)
SOA_CELL(8 , 256)
SOA_CELL(16, 256)
SOA_CELL(4 , 512)
SOA_CELL(8 , 512)
SOA_CELL(16, 512)

// initializer, which reads in matrices in matrix market format
// using mmc.h, mmc.c
template<typename CELL, typename GRID>
class SparseMatrixInitializerMM : public SimpleInitializer<CELL>
{
private:
    std::string fileName;
    int size;

public:
    inline
    SparseMatrixInitializerMM(
        const std::string& file,
        const Coord<3>& dim,
        int maxT) :
        SimpleInitializer<CELL>(Coord<1>(dim.x()), maxT),
        fileName(file),
        size(dim.x())
    {}

    virtual void grid(GridBase<CELL, 1> *ret)
    {
        // setup sparse matrix
        GRID *grid = dynamic_cast<GRID *>(ret);
        std::map<Coord<2>, double> adjacency;

        // read matrix into adjacency (this may take some time ...)
        // using this C API provided by matrix market
        MM_typecode matcode;
        int M, N, nz;
        FILE *f = fopen(fileName.c_str(), "r");
        if (!f) {
            throw std::logic_error("fopen() failed");
        }

        if (mm_read_banner(f, &matcode) != 0) {
            throw std::logic_error("Could not process Matrix Market banner");
        }

        if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
            throw std::logic_error("Could not dimensions of matrix");
        }

        if (size != M || size != N) {
            throw std::logic_error("Size mismatch");
        }

        for (int i = 0; i < nz; ++i) {
            int m, n;
            double tmp;
            fscanf(f, "%d %d %lg\n", &m, &n, &tmp);
            adjacency[Coord<2>(m - 1, n - 1)] = tmp;
        }

        fclose(f);

        grid->setAdjacency(0, adjacency);

        // setup rhs: not needed, since the grid is intialized with default cells
        // default value of SPMVMCell is 8.0
    }
};

template<typename CELL, std::string& FILENAME, int NZ, int C, int SIGMA>
class SparseMatrixVectorMultiplicationMM : public CPUBenchmark
{
private:
    typedef UnstructuredSoAGrid<CELL, 1, double, C, SIGMA> Grid;

    void updateFunctor(const Streak<1>& streak, const Grid& gridOld,
                       Grid *gridNew, unsigned nanoStep)
    {
        gridOld.callback(gridNew, UnstructuredUpdateFunctorHelpers::
                         UnstructuredGridSoAUpdateHelper<CELL>(
                             gridOld, gridNew, streak, nanoStep));
    }

public:
    virtual std::string family()
    {
        std::stringstream ss;
        ss << "SPMVM: C:" << C << " SIGMA:" << SIGMA;
        return ss.str();
    }

    virtual std::string species()
    {
        return FILENAME;
    }

    virtual double performance2(const Coord<3>& dim)
    {
        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializerMM<CELL, Grid> init(FILENAME, dim, maxT);
        init.grid(&gridOld);

        // 3. call updateFunctor()
        double seconds = 0;
        Streak<1> streak(Coord<1>(0), size.x());
        {
            ScopedTimer t(&seconds);
            updateFunctor(streak, gridOld, &gridNew, 0);
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * static_cast<double>(NZ);
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};

#ifdef __AVX__
template<typename CELL, std::string& FILENAME, int NZ, int C, int SIGMA>
class SparseMatrixVectorMultiplicationMMNative : public CPUBenchmark
{
private:
    typedef UnstructuredSoAGrid<CELL, 1, double, C, SIGMA> Grid;
    typedef SellCSigmaSparseMatrixContainer<double, C, SIGMA> Matrix;

    // callback to get cell's member pointer
    class GetPointer
    {
    private:
        double **sumPtr;
        double **valuePtr;
    public:
        GetPointer(double **sumPtr, double **valuePtr) :
            sumPtr(sumPtr), valuePtr(valuePtr)
        {}

        template<
            typename CELL1, long MY_DIM_X1, long MY_DIM_Y1, long MY_DIM_Z1, long INDEX1,
            typename CELL2, long MY_DIM_X2, long MY_DIM_Y2, long MY_DIM_Z2, long INDEX2>
        void operator()(
            LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& oldAccessor,
            LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& newAccessor) const
        {
            *sumPtr = &newAccessor.sum();
            *valuePtr = &oldAccessor.value();
        }
    };

public:
    std::string family()
    {
        std::stringstream ss;
        ss << "NATIVE: C:" << C << " SIGMA:" << SIGMA;
        return ss.str();
    }

    std::string species()
    {
        return FILENAME;
    }

    double performance2(const Coord<3>& dim)
    {
        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializerMM<CELL, Grid> init(FILENAME, dim, maxT);
        init.grid(&gridOld);

        // 3. native kernel
        const Matrix& matrix = gridOld.getAdjacency(0);
        const double *values = matrix.valuesVec().data();
        const int *cl = matrix.chunkLengthVec().data();
        const int *cs = matrix.chunkOffsetVec().data();
        const int *col = matrix.columnVec().data();
        double *rhsPtr; // = hoodOld.valuePtr;
        double *resPtr; // = hoodNew.sumPtr;
        gridOld.callback(&gridNew, GetPointer(&resPtr, &rhsPtr));
        const int rowsPadded = ((size.x() - 1) / C + 1) * C;
        double seconds = 0;
        {
            ScopedTimer t(&seconds);
            for (int i = 0; i < rowsPadded / C; ++i) {
                int offs = cs[i];
                __m256d tmp = _mm256_load_pd(resPtr + i*C);
                for (int j = 0; j < cl[i]; ++j) {
                    __m128d rhstmp;
                    __m256d rhs, val;
                    val    = _mm256_load_pd(values + offs);
                    rhstmp = _mm_loadl_pd(rhstmp, rhsPtr + col[offs++]);
                    rhstmp = _mm_loadh_pd(rhstmp, rhsPtr + col[offs++]);
                    rhs    = _mm256_insertf128_pd(rhs, rhstmp, 0);
                    rhstmp = _mm_loadl_pd(rhstmp, rhsPtr + col[offs++]);
                    rhstmp = _mm_loadh_pd(rhstmp, rhsPtr + col[offs++]);
                    rhs    = _mm256_insertf128_pd(rhs, rhstmp, 1);
                    tmp    = _mm256_add_pd(tmp, _mm256_mul_pd(val, rhs));
                }
                _mm256_store_pd(resPtr + i*C, tmp);
            }
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * static_cast<double>(NZ);
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};
#endif

std::string RM07 = "RM07R.mtx";
std::string KKT  = "kkt_power.mtx";
std::string HAM  = "Hamrle3.mtx";
std::string ML   = "ML_Geer.mtx";

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc > 4)) {
        std::cerr << "usage: " << argv[0] << " [-q,--quick] REVISION CUDA_DEVICE\n";
        return 1;
    }

    bool quick = false;
    int argumentIndex = 1;
    if (argc == 4) {
        if ((std::string(argv[1]) == "-q") ||
            (std::string(argv[1]) == "--quick")) {
            quick = true;
        }
        argumentIndex = 2;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    LibFlatArray::evaluate eval(revision);
    eval.print_header();

    // matrix: RM07R
    {
        const int NZ  = 37464962;
        const int DIM = 381689;

        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_1, RM07, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_1, RM07, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_1, RM07, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_2, RM07, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_2, RM07, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_2, RM07, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_4, RM07, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_4, RM07, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_4, RM07, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_16, RM07, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_16, RM07, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_16, RM07, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_32, RM07, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_32, RM07, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_32, RM07, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_64, RM07, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_64, RM07, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_64, RM07, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_128, RM07, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_128, RM07, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_128, RM07, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_256, RM07, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_256, RM07, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_256, RM07, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_512, RM07, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_512, RM07, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_512, RM07, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));

        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_1, RM07, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_1, RM07, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_1, RM07, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_2, RM07, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_2, RM07, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_2, RM07, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_4, RM07, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_4, RM07, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_4, RM07, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_16, RM07, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_16, RM07, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_16, RM07, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_32, RM07, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_32, RM07, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_32, RM07, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_64, RM07, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_64, RM07, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_64, RM07, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_128, RM07, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_128, RM07, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_128, RM07, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_256, RM07, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_256, RM07, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_256, RM07, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_512, RM07, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_512, RM07, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_512, RM07, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
    }

    // matrix: kkt_power
    {
        const int NZ  = 8130343;
        const int DIM = 2063494;

        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_1, KKT, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_1, KKT, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_1, KKT, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_2, KKT, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_2, KKT, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_2, KKT, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_4, KKT, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_4, KKT, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_4, KKT, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_16, KKT, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_16, KKT, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_16, KKT, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_32, KKT, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_32, KKT, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_32, KKT, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_64, KKT, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_64, KKT, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_64, KKT, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_128, KKT, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_128, KKT, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_128, KKT, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_256, KKT, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_256, KKT, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_256, KKT, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_512, KKT, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_512, KKT, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_512, KKT, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));

        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_1, KKT, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_1, KKT, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_1, KKT, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_2, KKT, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_2, KKT, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_2, KKT, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_4, KKT, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_4, KKT, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_4, KKT, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_16, KKT, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_16, KKT, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_16, KKT, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_32, KKT, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_32, KKT, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_32, KKT, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_64, KKT, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_64, KKT, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_64, KKT, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_128, KKT, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_128, KKT, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_128, KKT, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_256, KKT, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_256, KKT, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_256, KKT, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_512, KKT, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_512, KKT, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_512, KKT, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
    }

    // matrix: Hamrle3
    {
        const int NZ  = 5514242;
        const int DIM = 1447360;

        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_1, HAM, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_1, HAM, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_1, HAM, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_2, HAM, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_2, HAM, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_2, HAM, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_4, HAM, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_4, HAM, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_4, HAM, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_16, HAM, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_16, HAM, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_16, HAM, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_32, HAM, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_32, HAM, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_32, HAM, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_64, HAM, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_64, HAM, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_64, HAM, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_128, HAM, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_128, HAM, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_128, HAM, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_256, HAM, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_256, HAM, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_256, HAM, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_512, HAM, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_512, HAM, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_512, HAM, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));

        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_1, HAM, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_1, HAM, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_1, HAM, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_2, HAM, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_2, HAM, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_2, HAM, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_4, HAM, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_4, HAM, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_4, HAM, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_16, HAM, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_16, HAM, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_16, HAM, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_32, HAM, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_32, HAM, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_32, HAM, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_64, HAM, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_64, HAM, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_64, HAM, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_128, HAM, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_128, HAM, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_128, HAM, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_256, HAM, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_256, HAM, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_256, HAM, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_512, HAM, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_512, HAM, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_512, HAM, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
    }

    // matrix: ML_Geer
    {
        const int NZ  = 110879972;
        const int DIM = 1504002;

        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_1, ML, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_1, ML, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_1, ML, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_2, ML, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_2, ML, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_2, ML, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_4, ML, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_4, ML, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_4, ML, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_16, ML, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_16, ML, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_16, ML, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_32, ML, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_32, ML, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_32, ML, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_64, ML, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_64, ML, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_64, ML, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_128, ML, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_128, ML, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_128, ML, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_256, ML, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_256, ML, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_256, ML, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_4_512, ML, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_8_512, ML, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMM<SPMVMSoACell_16_512, ML, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));

        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_1, ML, NZ, 4, 1>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_1, ML, NZ, 8, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_1, ML, NZ, 16, 1>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_2, ML, NZ, 4, 2>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_2, ML, NZ, 8, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_2, ML, NZ, 16, 2>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_4, ML, NZ, 4, 4>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_4, ML, NZ, 8, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_4, ML, NZ, 16, 4>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_16, ML, NZ, 4, 16>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_16, ML, NZ, 8, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_16, ML, NZ, 16, 16>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_32, ML, NZ, 4, 32>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_32, ML, NZ, 8, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_32, ML, NZ, 16, 32>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_64, ML, NZ, 4, 64>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_64, ML, NZ, 8, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_64, ML, NZ, 16, 64>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_128, ML, NZ, 4, 128>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_128, ML, NZ, 8, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_128, ML, NZ, 16, 128>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_256, ML, NZ, 4, 256>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_256, ML, NZ, 8, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_256, ML, NZ, 16, 256>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_4_512, ML, NZ, 4, 512>(),
             toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_8_512, ML, NZ, 8, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
        // eval(SparseMatrixVectorMultiplicationMMNative<SPMVMSoACell_16_512, ML, NZ, 16, 512>(),
        //      toVector(Coord<3>(DIM, 1, 1)));
    }

    return 0;
}
