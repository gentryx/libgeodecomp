#include <emmintrin.h>
#include <mpi.h>

#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    typedef Topologies::Cube<3>::Topology Topology;

    static inline unsigned nanoSteps() 
    { 
        return 1; 
    }

    inline explicit Cell(const double& v=0) : temp(v)
    {}  

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[Coord<3>( 0,  0, -1)].temp + 
                neighborhood[Coord<3>( 0, -1,  0)].temp + 
                neighborhood[Coord<3>(-1,  0,  0)].temp + 
                neighborhood[Coord<3>( 1,  0,  0)].temp + 
                neighborhood[Coord<3>( 0,  1,  0)].temp + 
                neighborhood[Coord<3>( 0,  0,  1)].temp) * (1.0 / 6.0);
    }

    static void update(
        Cell *target, Cell* right, Cell *top, 
        Cell *center, Cell *bottom, Cell *left, 
        const int& length, const unsigned& nanoStep) 
    {
        double factor = 1.0 / 6.0;
        __m128d xFactor, cell1, cell2, cell3, cell4, tmp0, tmp1, tmp2, tmp3, tmp4;
        xFactor = _mm_set_pd(factor, factor);

        tmp0 = _mm_loadu_pd((double*) &center[-1]);

        for (int start = 0; start < length - 7; start +=8) {
            cell1 = _mm_load_pd((double*) &right[start] + 0);
            cell2 = _mm_load_pd((double*) &right[start] + 2);
            cell3 = _mm_load_pd((double*) &right[start] + 4);
            cell4 = _mm_load_pd((double*) &right[start] + 6);

            tmp1 = _mm_load_pd((double*) &top[start] + 0);
            tmp2 = _mm_load_pd((double*) &top[start] + 2);
            tmp3 = _mm_load_pd((double*) &top[start] + 4);
            tmp4 = _mm_load_pd((double*) &top[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_loadu_pd((double*) &bottom[start] + 1);
            tmp2 = _mm_loadu_pd((double*) &bottom[start] + 3);
            tmp3 = _mm_loadu_pd((double*) &bottom[start] + 5);
            tmp4 = _mm_loadu_pd((double*) &bottom[start] + 7);

            cell1 = _mm_add_pd(cell1, tmp0);
            cell2 = _mm_add_pd(cell2, tmp1);
            cell3 = _mm_add_pd(cell3, tmp2);
            cell4 = _mm_add_pd(cell4, tmp3);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp0 = tmp4;

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &left[start] + 0);
            tmp2 = _mm_load_pd((double*) &left[start] + 2);
            tmp3 = _mm_load_pd((double*) &left[start] + 4);
            tmp4 = _mm_load_pd((double*) &left[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            cell1 = _mm_mul_pd(cell1, xFactor);
            cell2 = _mm_mul_pd(cell2, xFactor);
            cell3 = _mm_mul_pd(cell3, xFactor);
            cell4 = _mm_mul_pd(cell4, xFactor);

            _mm_store_pd((double*) &target[start] + 0, cell1);
            _mm_store_pd((double*) &target[start] + 2, cell2);
            _mm_store_pd((double*) &target[start] + 4, cell3);
            _mm_store_pd((double*) &target[start] + 6, cell4);
        }
    }

    double temp;
};

//fixme: rework this interface
namespace LibGeoDecomp {

template<>
class UpdateFunctor<Cell> : public StreakUpdateFunctor<Cell>
{};

}

//LIBGEODECOMP_HAS_STREAK_UPDATE

// template<>
// class ProvidesStreakUpdate<Cell> : public boost::true_type 
// {};

class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    CellInitializer(int num) : SimpleInitializer<Cell>(Coord<3>(64, 64, 64 * num), 100)
    {}

    virtual void grid(GridBase<Cell, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        Coord<3> offset = 
            Coord<3>::diagonal(gridDimensions().x() * 5 / 128);
        int size = gridDimensions().x() * 50 / 128;
        

        for (int z = 0; z < size; ++z) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    Coord<3> c = offset + Coord<3>(x, y, z);
                    if (box.inBounds(c))
                        ret->at(c) = Cell(0.99999999999);
                }
            }
        }
    }
};

void runSimulation()
{
    int outputFrequency = 10;
    StripingSimulator<Cell> sim(
        new CellInitializer(MPILayer().size()),
        MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()), 
        1000,
        MPI::DOUBLE);
    if (MPILayer().rank() == 0)
        new TracingWriter<Cell>(&sim, outputFrequency);

    sim.run();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    runSimulation();
    MPI_Finalize();
    return 0;
}
