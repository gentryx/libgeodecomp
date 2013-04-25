#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/serialvisitwriter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/misc/commandserver.h>
#include <libgeodecomp/misc/dataaccessor.h>

#include <iostream>
#include <string>

using namespace boost::assign;
using namespace LibGeoDecomp;

/*
 *
 */
class ConwayCell
{
public:
    /*
     *
     */
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;

    /*
     *
     */
    class API : public CellAPITraits::Base
    {};

    /*
     *
     */
    static inline unsigned nanoSteps()
    {
        return 2;
    }

    /*
     *
     */
    ConwayCell(const bool& _alive = false) :
        alive(_alive)
    {
        if(alive)
            count = 1;
        else
            count = 0;
    }

    /*
     *
     */
    int countLivingNeighbors(const CoordMap<ConwayCell>& neighborhood)
    {
        int ret = 0;
        for (int y = -1; y < 2; ++y)
            for (int x = -1; x < 2; ++x)
                ret += neighborhood[Coord<2>(x, y)].alive;
        ret -= neighborhood[Coord<2>(0, 0)].alive;
        return ret;
    }

    /*
     *
     */
    void update(const CoordMap<ConwayCell>& neighborhood, const unsigned& nanoStep)
    {
        *this = neighborhood[Coord<2>(0, 0)];
        if (nanoStep == 0)
        {
            int livingNeighbors = countLivingNeighbors(neighborhood);
            alive = neighborhood[Coord<2>(0, 0)].alive;
            if (alive) {
                alive = (2 <= livingNeighbors) && (livingNeighbors <= 3);
            } else {
                alive = (livingNeighbors == 3);
            }
        }
        if (nanoStep == 1)
        {
            if(alive)
                count += 1;
        }
    }

    /*
     *
     */
    bool alive;
    long count;
};

/*
 *
 */
class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    /*
     *
     */
    CellInitializer() : SimpleInitializer<ConwayCell>(Coord<2>(160, 90), 800)
    {}

    /*
     *
     */
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

/*
 * define aliveDataAccessor
 */
DEFINE_DATAACCESSOR(ConwayCell, int, alive)

/*
 * define countDataAccessor
 */
DEFINE_DATAACCESSOR(ConwayCell, long, count)

/*
 * ---------------------------------------------
 * extend default remote steerer commands part 1
 * ---------------------------------------------
 */
struct mySteererData : SteererData<ConwayCell>
{
    mySteererData(DataAccessor<ConwayCell>** _dataAccessors, int _numVars) :
            SteererData(_dataAccessors, _numVars)
    {
    }
    boost::mutex size_mutex;
};

template<typename CELL_TYPE, int DIM>
class myControl : SteererControl<CELL_TYPE, DIM>
{
public:
    void operator()(typename Steerer<CELL_TYPE>::GridType *grid,
            const Region<Steerer<CELL_TYPE>::Topology::DIMENSIONS>& validRegion,
            const unsigned& step,
            CommandServer::Session* session,
            void *data)
    {
        mySteererData *sdata = (mySteererData*) data;
        if (sdata->size_mutex.try_lock())
        {
            std::string msg = "size: ";
            msg += boost::to_string(validRegion.size()) + "\n";
            session->sendMessage(msg);
        }
    }
};

static void sizeFunction(std::vector<std::string> stringVec,
        CommandServer::Session *session,
        void *data)
{
    mySteererData *sdata = (mySteererData*) data;
    std::string help_msg = "    Usage: size\n";
    help_msg += "          get the size of the region\n";
    if (stringVec.size() > 1)
    {
        session->sendMessage(help_msg);
        return;
    }
    sdata->size_mutex.unlock();
}

/*
 *
 */
void runSimulation()
{
    int outputFrequency = 1;

    DataAccessor<ConwayCell> *vars[3];
    vars[0] = new aliveDataAccessor();
    vars[1] = new countDataAccessor();

    SerialSimulator<ConwayCell> sim(new CellInitializer());

    sim.addWriter(
        new SerialVisitWriter<ConwayCell>(
        "./gameoflife_live",
        vars,
        2,
        outputFrequency)
    );

    /*
     * ---------------------------------------------
     * extend default remote steerer commands part 2
     * ---------------------------------------------
     */
    CommandServer::functionMap* fmap = RemoteSteerer<ConwayCell, 2>
            ::getDefaultMap();
    (*fmap)["size"] = sizeFunction;

    mySteererData* myData = new mySteererData(vars, 2);

    Steerer<ConwayCell>* steerer = new RemoteSteerer<ConwayCell, 2,
            DefaultSteererControl<ConwayCell, 2, myControl<ConwayCell, 2> > >
            (1, 1234, vars, 2, fmap, myData);

    sim.addSteerer(steerer);

    //sim.addSteerer(new RemoteSteerer<ConwayCell, 2>
    //        (1, 1234, vars, 2));

    sim.run();

    delete fmap;
    delete myData;
    for (int i = 0; i < 2; ++i)
    {
        delete vars[i];
    }
}

/*
 *
 */
int main(int argc, char *argv[])
{
    runSimulation();
    return 0;
}
