#include <mpi.h>

#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/visitwriter.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/misc/dataaccessor.h>
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/misc/remotesteererhelper.h>
#include <libgeodecomp/mpilayer/typemaps.h>

using namespace LibGeoDecomp;

enum State {
    FREE = 0,
    EAST = -1,
    SOUTH = 2,
    BLOCK = 1
};

/*
 *
 */
class Cell {
  public:
    /*
     *
     */
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;

    /*
     *
     */
    class API : public CellAPITraits::Base {};

    /*
     *
     */
    static inline unsigned nanoSteps() {
        return 2;
    }

    /*
     *
     */
    Cell(const int& _direction = FREE, const int& _border = 0, const int _rate = 5) :
        direction(_direction),
        border(_border),
        rate(_rate) {
    }

    /*
     *
     */
    void update(const CoordMap<Cell>& neighborhood, const unsigned& nanoStep) {
        *this = neighborhood[Coord<2>(0, 0)];
        if (nanoStep == 0) {
            if ((direction == FREE) && (neighborhood[Coord<2>(0, 1)].direction == SOUTH)) {
                direction = SOUTH;
            } else if ((direction == SOUTH) && (neighborhood[Coord<2>(0, -1)].direction == FREE)) {
                direction = FREE;
            }

            else if ((direction == FREE) && (neighborhood[Coord<2>(-1, -1)].direction == BLOCK)
                     && (neighborhood[Coord<2>(-1, 0)].direction == SOUTH)
                     && (neighborhood[Coord<2>(0, 1)].direction == FREE)) {
                direction = SOUTH;
            } else if ((direction == SOUTH) && (neighborhood[Coord<2>(0, -1)].direction == BLOCK)) {
                if ((neighborhood[Coord<2>(1, 0)].direction == FREE) &&
                    (neighborhood[Coord<2>(1, 1)].direction == FREE)) {
                    direction = FREE;
                }
            }

            else if ((direction == FREE) && (neighborhood[Coord<2>(-1, 0)].direction == EAST)
                && (neighborhood[Coord<2>(0, 1)].direction != SOUTH)) {
                direction = EAST;
            } else if ((direction == EAST) && (neighborhood[Coord<2>(1, 0)].direction == FREE)
                 && (neighborhood[Coord<2>(1, 1)].direction != SOUTH)) {
                direction = FREE;
            }

            else if ((direction == FREE) && (neighborhood[Coord<2>(1, 1)].direction == BLOCK)
                     && (neighborhood[Coord<2>(0, 1)].direction == EAST)
                     && (neighborhood[Coord<2>(-1, 0)].direction == FREE)) {
                direction = EAST;
            } else if ((direction == EAST) && (neighborhood[Coord<2>(1, 0)].direction == BLOCK)) {
                if ((neighborhood[Coord<2>(0, -1)].direction == FREE) &&
                    (neighborhood[Coord<2>(-1, -1)].direction == FREE)) {
                    direction = FREE;
                }
            }
        }
        if (nanoStep == 1) {
            if ((direction == FREE) && (border == 1)) {
                int random_integer = (rand()%100);;
                if (random_integer >= (100 - rate)) {
                    direction = SOUTH;
                }
            }
            if ((direction == FREE) && (border == -1)) {
                int random_integer = (rand()%100);;
                if (random_integer >= (100 - rate)) {
                    direction = EAST;
                }
            }
        }
    }

    /*
     *
     */
    int direction;
    int border;
    int rate;
};

/*
 *
 */
class CellInitializer : public SimpleInitializer<Cell> {
  public:
    /*
     *
     */
    CellInitializer() : SimpleInitializer<Cell>(Coord<2>(90, 90), 10000) {}

    /*
     *
     */
    virtual void grid(GridBase<Cell, 2> *ret) {
        for (int i = 1; i < 90; ++i) {
           ret->at(Coord<2>(i, 89)) = Cell(0, 1);
        }


        for (int j = 0; j < 89; ++j) {
            for (int i = 0; i < 90; ++i) {
                if (i == 0) {
                    ret->at(Coord<2>(i, j)) = Cell(0, -1);
                } else {
                    ret->at(Coord<2>(i, j)) = Cell(0);
                }
            }
        }

        for (int j = 43; j < 48; ++j) {
            for (int i = 43; i < 48; ++i) {
                ret->at(Coord<2>(i, j)).direction = BLOCK;
            }
        }

        for (int j = 23; j < 28; ++j) {
            for (int i = 73; i < 78; ++i) {
                ret->at(Coord<2>(i, j)).direction = BLOCK;
            }
        }
    }
};

DEFINE_DATAACCESSOR(Cell, int, border)
DEFINE_DATAACCESSOR(Cell, int, direction)
DEFINE_DATAACCESSOR(Cell, int, rate)

struct mySteererData : SteererData<Cell> {
    mySteererData(DataAccessor<Cell>** _dataAccessors, int _numVars) :
            SteererData(_dataAccessors, _numVars, MPI::COMM_WORLD) {
        getstep_mutex.lock();
    }
    boost::mutex getstep_mutex;
};

class MyControl : SteererControl<Cell, mySteererData> {
  public:
    void operator()(typename Steerer<Cell>::GridType *grid,
            const Region<Steerer<Cell>::Topology::DIM>& validRegion,
            const unsigned& step,
            RemoteSteererHelper::MessageBuffer* session,
            void *data,
            const MPI::Intracomm& _comm,
            bool _changed = true) {
        mySteererData* sdata = (mySteererData*) data;
        if (sdata->getstep_mutex.try_lock()) {
            std::string msg = "current step: ";
            msg += boost::to_string(step) + "\n";
            session->sendMessage(msg);
        }
    }
};

static void getStepFunction(std::vector<std::string> stringVec,
        CommandServer::Session *session,
        void *data) {
    mySteererData* sdata = (mySteererData*) data;
    std::string help_msg = "    Usage: getstep\n";
    help_msg += "          get the size of the region\n";
    if (stringVec.size() > 1) {
        if (stringVec.at(1).compare("help") == 0) {
            session->sendMessage(help_msg);
            return;
        }
    }
    std::string msg = "send step request\n";
    session->sendMessage(msg);
    sdata->getstep_mutex.unlock();
}

static void setRateFunction(std::vector<std::string> stringVec,
        CommandServer::Session *session,
        void *data) {
    std::string help_msg = "    Usage: setrate <direction> <probability>\n";
    help_msg += "          sets the probability of a new car for the given direction\n";
    help_msg += "          directions: east and south\n";
    help_msg += "          probability: int between 0 and 100\n";
    std::cout << "called setRateFunction" << std::endl;
    if (stringVec.size() != 3) {
        std::cout << "wrong number" << std::endl;
        session->sendMessage(help_msg);
        return;
    }
    int value;
    try {
        value = mystrtoi(stringVec.at(2).c_str());
    }
    catch (std::exception& e) {
        std::cout << "toi fehler" << std::endl;
        session->sendMessage(help_msg);
        return;
    }
    if ((value < 0) || (value > 100)) {
        std::cout << "range fehler" << std::endl;
        session->sendMessage(help_msg);
        session->sendMessage(help_msg);
        return;
    }
    if (stringVec.at(1).compare("east") == 0) {
        for (int i = 0; i < 89; ++i) {
            std::vector<std::string> newCommandVec;
            newCommandVec.push_back("set");
            newCommandVec.push_back("rate");
            newCommandVec.push_back(stringVec.at(2));
            newCommandVec.push_back("0");
            newCommandVec.push_back(boost::to_string(i));
            RemoteSteerer<Cell>::setFunction(newCommandVec, session, data);
        }
    } else if (stringVec.at(1).compare("south") == 0) {
        for (int i = 1; i < 90; ++i) {
            std::vector<std::string> newCommandVec;
            newCommandVec.push_back("set");
            newCommandVec.push_back("rate");
            newCommandVec.push_back(stringVec.at(2));
            newCommandVec.push_back(boost::to_string(i));
            newCommandVec.push_back("89");
            RemoteSteerer<Cell>::setFunction(newCommandVec, session, data);
        }
    } else {
        std::cout << "direction falsch" << std::endl;
        session->sendMessage(help_msg);
        return;
    }
    std::string msg = "send setrate request\n";
    std::cout << "setrate finish" << std::endl;
    session->sendMessage(msg);
}

/*
 *
 */
void runSimulation() {
    DataAccessor<Cell> *vars[3];
    vars[0] = new borderDataAccessor();
    vars[1] = new directionDataAccessor();
    vars[2] = new rateDataAccessor();

    SerialSimulator<Cell> sim(new CellInitializer());

    sim.addWriter(
        new VisitWriter<Cell>("./cars", vars, 3, 1, VISIT_SIMMODE_STOPPED));

    sim.addWriter(
        new TracingWriter<Cell>(
            1,
            10000));

    mySteererData* myData = new mySteererData(vars, 3);

    CommandServer::functionMap* fmap = RemoteSteerer<Cell>::getDefaultMap();
    (*fmap)["getstep"] = getStepFunction;
    (*fmap)["setrate"] = setRateFunction;

    Steerer<Cell>* steerer = new RemoteSteerer <Cell, mySteererData,
        DefaultSteererControl<Cell, mySteererData, MyControl> >(
        1, 1234, vars, 3, fmap, (void*)myData, MPI::COMM_WORLD);
    sim.addSteerer(steerer);

    sim.run();

    for (int i = 0; i < 2; ++i) {
        delete vars[i];
    }
}

/*
 *
 */
int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);
    srand((unsigned)time(0));
    runSimulation();
    MPI::Finalize();
    return 0;
}
