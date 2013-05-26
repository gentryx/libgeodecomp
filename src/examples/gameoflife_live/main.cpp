#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/visitwriter.h>
#include <libgeodecomp/io/remotesteerer/commandserver.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

class ConwayCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;

    class API : public CellAPITraits::Base
    {};

    static inline unsigned nanoSteps()
    {
        return 1;
    }

    ConwayCell(const bool& _alive = false) :
        alive(_alive)
    {
        count = alive? 1 : 0;
    }

    int countLivingNeighbors(const CoordMap<ConwayCell>& neighborhood)
    {
        int ret = 0;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                ret += neighborhood[Coord<2>(x, y)].alive;
            }
        }
        ret -= neighborhood[Coord<2>(0, 0)].alive;
        return ret;
    }

    void update(const CoordMap<ConwayCell>& neighborhood, const unsigned&)
    {
        int livingNeighbors = countLivingNeighbors(neighborhood);
        alive = neighborhood[Coord<2>(0, 0)].alive;
        if (alive) {
            alive = (2 <= livingNeighbors) && (livingNeighbors <= 3);
        } else {
            alive = (livingNeighbors == 3);
        }
        if(alive)
            count += 1;
    }

    bool alive;
    int count;
};

class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    CellInitializer() : SimpleInitializer<ConwayCell>(Coord<2>(160, 90), 800)
    {}

    virtual void grid(GridBase<ConwayCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        SuperVector<Coord<2> > startCells;
        // start with a single glider...
        //          x
        //           x
        //         xxx
        startCells +=
            Coord<2>(11, 10),
            Coord<2>(12, 11),
            Coord<2>(10, 12), Coord<2>(11, 12), Coord<2>(12, 12);


        // ...add a Diehard pattern...
        //                x
        //          xx
        //           x   xxx
        startCells +=
            Coord<2>(55, 70), Coord<2>(56, 70), Coord<2>(56, 71), 
            Coord<2>(60, 71), Coord<2>(61, 71), Coord<2>(62, 71), 
            Coord<2>(61, 69);

        // ...and an Acorn pattern:
        //        x 
        //          x
        //       xx  xxx
        startCells +=
            Coord<2>(111, 30),
            Coord<2>(113, 31),
            Coord<2>(110, 32), Coord<2>(111, 32), 
            Coord<2>(113, 32), Coord<2>(114, 32), Coord<2>(115, 32);

        for (SuperVector<Coord<2> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i) 
            if (rect.inBounds(*i))
                ret->at(*i - rect.origin) = ConwayCell(true);
    }
};

DEFINE_DATAACCESSOR(ConwayCell, char, alive);
DEFINE_DATAACCESSOR(ConwayCell, int, count);

class MySteererData : public SteererData<ConwayCell>
{
public:
    MySteererData() :
        SteererData<ConwayCell>()
    {}

    boost::mutex size_mutex;
};

template<typename CELL_TYPE, typename DATATYPE>
class MyControl : RemoteSteererHelper::SteererControl<CELL_TYPE, DATATYPE>
{
public:
    virtual void operator()(
        typename Steerer<CELL_TYPE>::GridType *grid,
        const Region<Steerer<CELL_TYPE>::Topology::DIM>& validRegion,
        const unsigned& step,
        MessageBuffer *session,
        DATATYPE *data,
        const MPI::Intracomm& comm,
        bool changed)
    {
        MySteererData *sdata = (MySteererData*)data;
        if (sdata->size_mutex.try_lock()) {
            std::string msg = "size: ";
            msg += boost::to_string(validRegion.size()) + "\n";
            session->sendMessage(msg);
        }
    }
};

// fixme: replace static functions with fuctors
static void sizeFunction(std::vector<std::string> stringVec,
        CommandServer::Session *session,
        void *data)
{
    MySteererData *sdata = (MySteererData*)data;
    std::string help_msg = "    Usage: size\n";
    help_msg += "          get the size of the region\n";
    if (stringVec.size() > 1) {
        session->sendMessage(help_msg);
        return;
    }
    sdata->size_mutex.unlock();
}

void runSimulation()
{
    int outputFrequency = 10;

    SerialSimulator<ConwayCell> sim(
        new CellInitializer());


    VisItWriter<ConwayCell> *visItWriter = new VisItWriter<ConwayCell>(
        "gameOfLife", outputFrequency, VISIT_SIMMODE_STOPPED);
    visItWriter->addVariable(new aliveDataAccessor());
    visItWriter->addVariable(new countDataAccessor());

    sim.addWriter(visItWriter);

    /*
     * ---------------------------------------------
     * extend default remote steerer commands part 2
     * ---------------------------------------------
     */
    CommandServer::FunctionMap functionMap = RemoteSteerer<ConwayCell>::getDefaultMap();
    functionMap["size"] = sizeFunction;

    MySteererData *myData = new MySteererData();
    myData->addVariable(new aliveDataAccessor());
    myData->addVariable(new countDataAccessor());

    // fixme: class names must start with capitals
    // fixme: too long template instantiation

    Steerer<ConwayCell> *steerer =
        new RemoteSteerer<ConwayCell,
                          MySteererData,
                          DefaultSteererControl<
                              ConwayCell, MySteererData, MyControl<
                                  ConwayCell, SteererData<ConwayCell> > > >(
                                      1,
                                      1234,
                                      functionMap,
                                      myData);
    sim.addSteerer(steerer);

    sim.run();
}

int main(int argc, char *argv[])
{
    runSimulation();
    return 0;
}
