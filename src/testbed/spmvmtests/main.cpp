/**
 * This test operates on the following matrices:
 *   - http://www.cise.ufl.edu/research/sparse/MM/Fluorem/RM07R.tar.gz
 *   - http://www.cise.ufl.edu/research/sparse/MM/Zaoui/kkt_power.tar.gz
 *   - http://www.cise.ufl.edu/research/sparse/MM/Janna/ML_Geer.tar.gz
 *   - http://www.cise.ufl.edu/research/sparse/MM/Hamrle/Hamrle3.tar.gz
 *
 * Use the accompanying fetch_matrices.sh to download/extract these.
 *
 * We're using the Matrix Market IO library for matrix parsing:
 * http://math.nist.gov/MatrixMarket/mmio-c.html
 *
 */
#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>
#include <libgeodecomp/storage/unstructuredupdatefunctor.h>
#include <libgeodecomp/testbed/spmvmtests/mmio.h>

#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/macros.hpp>

#if defined(__AVX__) || defined(__MIC__)
#include <immintrin.h>
#endif

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <map>

using namespace LibGeoDecomp;
using namespace LibFlatArray;

#ifdef __AVX__
static const int C = 4;
#endif
#ifdef __MIC__
static const int C = 16;
#endif

#if defined(__AVX__) && defined(__MIC__)
# error "AVX _and_ KNC enabled which makes no sense"
#endif

// without vectorization, at least the LGD can be tested
// since it uses short_vecs
#if !defined(__AVX__) && !defined(__MIC__)
static const int C = 4;
#endif

template<int SIGMA>
class SPMVMSoACell
{
public:
    class API :
        public APITraits::HasSoA,
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES(
            (16)(32)(64)(128)(256)(512)(1024)(2048)(4096)(8192)(16384)(32768)
            (65536)(131072)(262144)(524288)(1048576)(2097152)(4194304),
            (1),
            (1))
    };

    typedef short_vec<double, C> ShortVec;

    inline explicit SPMVMSoACell(double v = 8.0) :
        value(v), sum(0)
    {}

#ifdef __MIC__
    /**
     * Does the same as below, but is written a little bit different.
     * By this rewriting the ICC is able to insert prefetching instructions
     * which are needed for performance.
     */
    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        ShortVec tmp, weights, values;
        auto indPtr = (*hoodOld.begin()).first;
        auto matPtr = (*hoodOld.begin()).second;
        auto sumPtr = &hoodNew->sum();
        int offs = 0;
#pragma loop count(1000)
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            tmp.load_aligned(sumPtr + i * C);
            for (const auto& j: hoodOld.weights(0)) {
                weights.load_aligned(matPtr + offs);
                values.gather(&hoodOld->value(), indPtr + offs);
                tmp += values * weights;
                offs += C;
            }
            tmp.store_aligned(sumPtr + i * C);
        }
    }
#else
    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        ShortVec tmp, weights, values;
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            tmp.load_aligned(&hoodNew->sum() + i * C);
            for (const auto& j: hoodOld.weights(0)) {
                weights.load_aligned(j.second);
                values.gather(&hoodOld->value(), j.first);
                tmp += values * weights;
            }
            tmp.store_aligned(&hoodNew->sum() + i * C);
        }
    }
#endif

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    double value;
    double sum;
};

// multiple LIBFLATARRAY_REGISTER_SOA are needed right now, see:
// https://bitbucket.org/gentryx/libflatarray/issue/16/libflatarray_register_soa-does-not-work
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<1>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<4>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<32>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<64>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<128>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<512>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<4096>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<8192>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<16384>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<32768>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<65536>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<131072>, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell<262144>, ((double)(sum))((double)(value)))

#define SPMVM_TESTS(METHOD, MATRIX)                                     \
    do {                                                                \
        eval(METHOD<SPMVMSoACell<1     >, MATRIX, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<4     >, MATRIX, NZ, 4>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<32    >, MATRIX, NZ, 32>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<64    >, MATRIX, NZ, 64>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<128   >, MATRIX, NZ, 128>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<512   >, MATRIX, NZ, 512>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<4096  >, MATRIX, NZ, 4096>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<8192  >, MATRIX, NZ, 8192>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<16384 >, MATRIX, NZ, 16384>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<32768 >, MATRIX, NZ, 32768>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<65536 >, MATRIX, NZ, 65536>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<131072>, MATRIX, NZ, 131072>(), toVector(Coord<3>(DIM, 1, 1))); \
        eval(METHOD<SPMVMSoACell<262144>, MATRIX, NZ, 262144>(), toVector(Coord<3>(DIM, 1, 1))); \
    } while (0)

/**
 * For reference performance of SELL, we also measure the performance of
 * compressed row storage. This class initializes the datastructures needed
 * for CRS.
 */
template<typename VALUE_TYPE>
class CRSInitializer
{
private:
    int dimension;
    std::vector<VALUE_TYPE> values;
    std::vector<int>        column;
    std::vector<int>        rowLen;

    void initFromMatrix(const std::map<Coord<2>, VALUE_TYPE>& matrix)
    {
        const int nonZero = matrix.size();
        if (!nonZero) {
            throw std::logic_error("Matrix should at least have one non-zero entry");
        }

        values.resize(nonZero);
        column.resize(nonZero);
        rowLen.resize(dimension + 1);

        int currentRow = 0;
        int cnt = 0;
        rowLen[0] = 0;
        for (const auto& pair : matrix) {
            if (currentRow != pair.first.x()) {
                currentRow = pair.first.x();
                rowLen[currentRow] = cnt;
            }
            values[cnt] = pair.second;
            column[cnt] = pair.first.y();
            ++cnt;
        }
        rowLen[dimension] = cnt;
    }

public:
    inline
    explicit CRSInitializer(int dim) :
        dimension(dim)
    {}

    inline
    const std::vector<VALUE_TYPE>& valuesVec() const
    {
        return values;
    }

    inline
    const std::vector<int>& columnVec() const
    {
        return column;
    }

    inline
    const std::vector<int>& rowLenVec() const
    {
        return rowLen;
    }

    void init(const std::string& fileName)
    {
        std::map<Coord<2>, VALUE_TYPE> adjacency;

        // read matrix into adjacency (this may take some time ...)
        // using this C API provided by matrix market
        MM_typecode matcode;
        int M, N, nz;
        FILE *f = fopen(fileName.c_str(), "r");
        if (!f) {
            LOG(FATAL, "CRSInitializer failed to open file " << fileName);
            throw std::logic_error("fopen() failed");
        }

        if (mm_read_banner(f, &matcode) != 0) {
            throw std::logic_error("Could not process Matrix Market banner");
        }

        if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
            throw std::logic_error("Could not dimensions of matrix");
        }

        if (dimension != M || dimension != N) {
            throw std::logic_error("Size mismatch");
        }

        for (int i = 0; i < nz; ++i) {
            int m, n;
            double tmp;
            if (fscanf(f, "%d %d %lg\n", &m, &n, &tmp) != 3) {
                throw std::logic_error("Failed to parse mtx format");
            }
            adjacency[Coord<2>(m - 1, n - 1)] = tmp;
        }

        fclose(f);

        initFromMatrix(adjacency);
    }
};

/**
 * Initializer class, which reads in matrices in matrix market format
 * using mmc.h, mmc.c. See http://math.nist.gov/MatrixMarket/mmio-c.html.
 */
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

    virtual void grid(GridBase<CELL, 1> *grid)
    {
        // setup sparse matrix
        std::map<Coord<2>, double> weights;

        // read matrix into weights (this may take some time ...)
        // using this C API provided by matrix market
        MM_typecode matcode;
        int M, N, nz;
        FILE *f = fopen(fileName.c_str(), "r");
        if (!f) {
            LOG(FATAL, "SparseMatrixInitializerMM failed to open file " << fileName);
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
            if (fscanf(f, "%d %d %lg\n", &m, &n, &tmp) != 3) {
                throw std::logic_error("Failed to parse mtx format");
            }
            weights[Coord<2>(m - 1, n - 1)] = tmp;
        }

        fclose(f);

        grid->setWeights(0, weights);

        // setup rhs: not needed, since the grid is intialized with default cells
        // default value of SPMVMCell is 8.0
    }
};

template<typename CELL, std::string& FILENAME, int NZ, int SIGMA>
class SparseMatrixVectorMultiplicationMM : public CPUBenchmark
{
private:
    typedef UnstructuredSoAGrid<CELL, 1, double, C, SIGMA> Grid;

    void updateFunctor(const Region<1>& region, const Grid& gridOld,
                       Grid *gridNew, unsigned nanoStep)
    {
        gridOld.callback(
            gridNew,
            UnstructuredUpdateFunctorHelpers::UnstructuredGridSoAUpdateHelper<CELL>(
                gridOld, gridNew, region, nanoStep));
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

    virtual double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
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
        Region<1> region;
        region << Streak<1>(Coord<1>(0), size.x());
        {
            ScopedTimer t(&seconds);
            updateFunctor(region, gridOld, &gridNew, 0);
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
template<typename CELL, std::string& FILENAME, int NZ, int SIGMA>
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

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializerMM<CELL, Grid> init(FILENAME, dim, maxT);
        init.grid(&gridOld);

        // 3. native kernel
        const Matrix& matrix = gridOld.getWeights(0);
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
                    __m256d rhs;
                    __m256d val;
                    val    = _mm256_load_pd(values + offs);
                    rhs = _mm256_set_pd(
                        *(rhsPtr + col[j + 3]),
                        *(rhsPtr + col[j + 2]),
                        *(rhsPtr + col[j + 1]),
                        *(rhsPtr + col[j + 0]));
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

#ifdef __MIC__
template<typename CELL, std::string& FILENAME, int NZ, int SIGMA>
class SparseMatrixVectorMultiplicationMMNativeMIC : public CPUBenchmark
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

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializerMM<CELL, Grid> init(FILENAME, dim, maxT);
        init.grid(&gridOld);

        // 3. native kernel
        const Matrix& matrix = gridOld.getWeights(0);
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
            __m512d tmp1, tmp2, val, rhs;
            __m512i idx;
            int offs;
            for (int i = 0; i < rowsPadded / C; ++i) {
                tmp1 = _mm512_load_pd(resPtr + i*C + 0);
                tmp2 = _mm512_load_pd(resPtr + i*C + 8);
                offs = cs[i];
                for (int j = 0; j < cl[i]; ++j) {
                    val  = _mm512_load_pd(values + offs);
                    idx  = _mm512_load_epi32(col + offs);
                    rhs  = _mm512_i32logather_pd(idx, rhsPtr, 8);
                    tmp1 = _mm512_add_pd(tmp1, _mm512_mul_pd(val, rhs));
                    offs += 8;

                    val  = _mm512_load_pd(values + offs);
                    idx  = _mm512_permute4f128_epi32(idx, _MM_PERM_BADC);
                    rhs  = _mm512_i32logather_pd(idx, rhsPtr, 8);
                    tmp2 = _mm512_add_pd(tmp2, _mm512_mul_pd(val, rhs));
                    offs += 8;
                }
                _mm512_store_pd(resPtr + i*C + 0, tmp1);
                _mm512_store_pd(resPtr + i*C + 8, tmp2);
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

#ifdef __AVX__
template<typename CELL, std::string& FILENAME, int NZ, int SIGMA>
class SparseMatrixVectorMultiplicationMMCRS : public CPUBenchmark
{
private:
    typedef UnstructuredSoAGrid<CELL, 1, double, C, SIGMA> Grid;

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
        ss << "CRS";
        return ss.str();
    }

    std::string species()
    {
        return FILENAME;
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init crs data structures
        CRSInitializer<double> crsInit(dim.x());
        crsInit.init(FILENAME);

        // 3. native kernel
        const double *values = crsInit.valuesVec().data();
        const int *col = crsInit.columnVec().data();
        const int *row = crsInit.rowLenVec().data();
        double *rhsPtr; // = hoodOld.valuePtr;
        double *resPtr; // = hoodNew.sumPtr;
        gridOld.callback(&gridNew, GetPointer(&resPtr, &rhsPtr));
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            // AVX code
            for (int i = 0; i < dim.x(); ++i) {
                int j;
                __m256d sum;
                __m256d rhs;
                __m256d val;
                sum = _mm256_setzero_pd();
                for (j = row[i]; j < row[i + 1] - 3; j += 4) {
                    val = _mm256_loadu_pd(values + j);
                    rhs = _mm256_set_pd(
                        *(rhsPtr + col[j + 3]),
                        *(rhsPtr + col[j + 2]),
                        *(rhsPtr + col[j + 1]),
                        *(rhsPtr + col[j + 0]));
                    sum    = _mm256_add_pd(sum, _mm256_mul_pd(val, rhs));
                }
                double tmp[4] __attribute__((aligned (32)));
                _mm256_store_pd(tmp, sum);
                resPtr[i] = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                // reminder loop
                for (; j < row[i + 1]; ++j) {
                    resPtr[i] += values[j] * rhsPtr[col[j]];
                }
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

#ifdef __MIC__
template<typename CELL, std::string& FILENAME, int NZ, int SIGMA>
class SparseMatrixVectorMultiplicationMMCRSMIC : public CPUBenchmark
{
private:
    typedef UnstructuredSoAGrid<CELL, 1, double, C, SIGMA> Grid;

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
        ss << "CRS";
        return ss.str();
    }

    std::string species()
    {
        return FILENAME;
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        // 1. create grids
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init crs data structures
        CRSInitializer<double> crsInit(dim.x());
        crsInit.init(FILENAME);

        // 3. native kernel
        const double *values = crsInit.valuesVec().data();
        const int *col = crsInit.columnVec().data();
        const int *row = crsInit.rowLenVec().data();
        double *rhsPtr; // = hoodOld.valuePtr;
        double *resPtr; // = hoodNew.sumPtr;
        gridOld.callback(&gridNew, GetPointer(&resPtr, &rhsPtr));
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            // MIC code
            for (int i = 0; i < dim.x(); ++i) {
                int j;
                __m512d sum;
                __m512d rhs;
                __m512d val;
                __m512i idx;
                sum = _mm512_setzero_pd();
                for (j = row[i]; j < row[i + 1] - 7; j += 8) {
                    val = _mm512_loadunpacklo_pd(val, values + 0);
                    val = _mm512_loadunpackhi_pd(val, values + 8);
                    idx = _mm512_loadunpacklo_epi32(idx, col + 0);
                    rhs = _mm512_i32logather_pd(idx, rhsPtr, 8);
                    sum = _mm512_add_pd(sum, _mm512_mul_pd(val, rhs));
                }
                resPtr[i] = _mm512_reduce_add_pd(sum);

                // reminder loop
                for (; j < row[i + 1]; ++j) {
                    resPtr[i] += values[j] * rhsPtr[col[j]];
                }
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
    if ((argc < 3) || (argc == 4) || (argc > 5)) {
        std::cerr << "usage: " << argv[0] << " [-n,--name SUBSTRING] REVISION CUDA_DEVICE \n"
                  << "  - optional: only run tests whose name contains a SUBSTRING,\n"
                  << "  - REVISION is purely for output reasons,\n"
                  << "  - CUDA_DEVICE causes CUDA tests to run on the device with the given ID.\n";
        return 1;
    }
    std::string name = "";
    int argumentIndex = 1;
    if (argc == 5) {
        if ((std::string(argv[1]) == "-n") ||
            (std::string(argv[1]) == "--name")) {
            name = std::string(argv[2]);
        }
        argumentIndex = 3;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    evaluate eval(name, revision);
    eval.print_header();

    // matrix: RM07R
    {
        const int NZ  = 37464962;
        const int DIM = 381689;

        // SPMVM_TESTS(SparseMatrixVectorMultiplicationMM, RM07);

#ifdef __AVX__
        // SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNative, RM07);
        eval(SparseMatrixVectorMultiplicationMMCRS<SPMVMSoACell<1>, RM07, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif

#ifdef __MIC__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNativeMIC, RM07);
        eval(SparseMatrixVectorMultiplicationMMCRSMIC<SPMVMSoACell<1>, RM07, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif
    }

    // matrix: kkt_power
    {
        const int NZ  = 8130343;
        const int DIM = 2063494;

        SPMVM_TESTS(SparseMatrixVectorMultiplicationMM, KKT);

#ifdef __AVX__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNative, KKT);
        eval(SparseMatrixVectorMultiplicationMMCRS<SPMVMSoACell<1>, KKT, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif

#ifdef __MIC__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNativeMIC, KKT);
        eval(SparseMatrixVectorMultiplicationMMCRSMIC<SPMVMSoACell<1>, KKT, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif
    }

    // matrix: Hamrle3
    {
        const int NZ  = 5514242;
        const int DIM = 1447360;

        SPMVM_TESTS(SparseMatrixVectorMultiplicationMM, HAM);

#ifdef __AVX__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNative, HAM);
        eval(SparseMatrixVectorMultiplicationMMCRS<SPMVMSoACell<1>, HAM, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif

#ifdef __MIC__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNativeMIC, HAM);
        eval(SparseMatrixVectorMultiplicationMMCRSMIC<SPMVMSoACell<1>, HAM, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif
    }

    // matrix: ML_Geer
    {
        const int NZ  = 110879972;
        const int DIM = 1504002;

        SPMVM_TESTS(SparseMatrixVectorMultiplicationMM, ML);

#ifdef __AVX__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNative, ML);
        eval(SparseMatrixVectorMultiplicationMMCRS<SPMVMSoACell<1>, ML, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif

#ifdef __MIC__
        SPMVM_TESTS(SparseMatrixVectorMultiplicationMMNativeMIC, ML);
        eval(SparseMatrixVectorMultiplicationMMCRSMIC<SPMVMSoACell<1>, ML, NZ, 1>(), toVector(Coord<3>(DIM, 1, 1)));
#endif
    }

    return 0;
}
