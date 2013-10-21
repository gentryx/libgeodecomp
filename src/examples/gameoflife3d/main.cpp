#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/visitwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <boost/assign/std/vector.hpp>

#include <libgeodecomp/storage/grid.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

class ConwayCell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    ConwayCell(const bool& alive = false) :
        alive(alive)
    {}

    template<typename COORD_MAP>
    int countLivingNeighbors(const COORD_MAP& neighborhood)
    {
        int ret = 0;
        for (int z = -1; z < 2; ++z) {
            for (int y = -1; y < 2; ++y) {
                for (int x = -1; x < 2; ++x) {
                    ret += neighborhood[Coord<3>(x, y, z)].alive;
                }
            }
        }
        ret -= neighborhood[Coord<3>(0, 0, 0)].alive;
        return ret;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned&)
    {
        int livingNeighbors = countLivingNeighbors(neighborhood);
        alive = neighborhood[Coord<3>(0, 0, 0)].alive;
        if (alive) {
            alive = (5 <= livingNeighbors) && (livingNeighbors <= 7);
        } else {
            alive = (livingNeighbors == 6);
        }
    }

    bool alive;
};

class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    CellInitializer() : SimpleInitializer<ConwayCell>(Coord<3>(20, 20, 20), 800)
    {}

    virtual void grid(GridBase<ConwayCell, 3> *ret)
    {
        CoordBox<3> rect = ret->boundingBox();
        std::vector<Coord<3> > startCells;
        int tmp;
        for (int z = 0; z < 20; ++z) {
            for (int y = 0; y < 20; ++y) {
                for (int x = 0; x < 20; ++x) {
                    tmp = (x+y+z) % 2;
                    if (tmp) {
                        if (rect.inBounds(Coord<3>(x, y, z))) {
                            startCells += Coord<3>(x, y, z);
                        }
                    }
                }
            }
        }

        for (std::vector<Coord<3> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i)
            {
            if (rect.inBounds(*i)) {
                ret->set(*i, ConwayCell(true));
            }
        }
    }
};

DEFINE_DATAACCESSOR(AliveAccessor, ConwayCell, char, alive)

void runSimulation()
{
    int outputFrequency = 1;

    SerialSimulator<ConwayCell> sim(new CellInitializer());

    VisItWriter<ConwayCell> *visItWriter = new VisItWriter<ConwayCell>(
        "gameOfLife3D",
        outputFrequency,
        VISIT_SIMMODE_STOPPED);
    visItWriter->addVariable(new AliveAccessor());

    sim.addWriter(visItWriter);

    RemoteSteerer<ConwayCell> *steerer = new RemoteSteerer<ConwayCell>(1, 1234);
    steerer->addDataAccessor(new AliveAccessor());
    sim.addSteerer(steerer);

    sim.run();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    LibGeoDecomp::Typemaps::initializeMaps();

    runSimulation();

    MPI_Finalize();
    return 0;
}
