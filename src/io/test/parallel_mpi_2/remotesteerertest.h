#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RemoteSteererTest : public CxxTest::TestSuite
{
public:
    class FlushAction : public Action<TestCell<2> >
    {
    public:
        FlushAction() :
            Action<TestCell<2> >(
                "flush",
                "Usage: \"flush VALUE TIME\"\nWill add VALUE to all TestCell.testValue at time step TIME")
        {}

        virtual void operator()(const StringVec& parameters, Pipe& pipe)
        {
            pipe.addSteeringRequest("flush " + parameters[0] + " " + parameters[1]);
            pipe.addSteeringFeedback("flush received");
        }
    };

    class FlushHandler : public Handler<TestCell<2> >
    {
    public:
        FlushHandler() :
            Handler<TestCell<2> >("flush")
        {}

        virtual bool operator()(const StringVec& parameters, Pipe& pipe, GridType *grid, const Region<Topology::DIM>& validRegion, unsigned step)
        {
            std::stringstream buf;
            buf << parameters[0] << " " << parameters[1];
            double value;
            double requestedStep;
            buf >> value;
            buf >> requestedStep;

            if (step != requestedStep) {
                return false;
            }

            for (Region<2>::Iterator i = validRegion.begin();
                 i != validRegion.end();
                 ++i) {
                grid->at(*i).testValue += value;
            }

            return true;
        }

    };

    void setUp()
    {
        unsigned steererPeriod = 3;
        unsigned writerPeriod = 2;
        int port = 47112;
        int maxSteps = 30;

        sim.reset(new StripingSimulator<TestCell<2> >(
                      new TestInitializer<TestCell<2> >(Coord<2>(20, 10), maxSteps),
                      mpiLayer.rank() ? 0 : new NoOpBalancer,
                      10000000));

        steerer = new RemoteSteerer<TestCell<2> >(steererPeriod, port);
        steerer->addHandler(new FlushHandler);

        writer = new ParallelMemoryWriter<TestCell<2> >(writerPeriod);
        sim->addSteerer(steerer);
        sim->addWriter(writer);
    }

    void tearDown()
    {
        sim.reset();
    }

    void testBasic()
    {
        LOG(DEBUG, "bongoA " << MPILayer().rank() << ")");
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new FlushAction);
            StringVec feedback = steerer->sendCommandWithFeedback("flush 1234 9", 1);
            TS_ASSERT_EQUALS(1, feedback.size());
            TS_ASSERT_EQUALS("flush received", feedback[0]);
        }

        LOG(DEBUG, "bongoB " << MPILayer().rank() << ")");
        sim->run();
        LOG(DEBUG, "bongoC " << MPILayer().rank() << ")");

        TS_ASSERT_EQUALS(writer->getGrids().size(), 16);
        TS_ASSERT_EQUALS(writer->getGrid( 0)[Coord<2>(3, 3)].testValue, 64.0);
        TS_ASSERT_EQUALS(writer->getGrid( 0)[Coord<2>(4, 3)].testValue, 65.0);

        TS_ASSERT_EQUALS(writer->getGrid( 2)[Coord<2>(3, 3)].testValue, 64.0);
        TS_ASSERT_EQUALS(writer->getGrid( 2)[Coord<2>(4, 3)].testValue, 65.0);

        TS_ASSERT_EQUALS(writer->getGrid( 4)[Coord<2>(3, 3)].testValue, 64.0);
        TS_ASSERT_EQUALS(writer->getGrid( 4)[Coord<2>(4, 3)].testValue, 65.0);

        TS_ASSERT_EQUALS(writer->getGrid( 6)[Coord<2>(3, 3)].testValue, 64.0);
        TS_ASSERT_EQUALS(writer->getGrid( 6)[Coord<2>(4, 3)].testValue, 65.0);

        TS_ASSERT_EQUALS(writer->getGrid( 8)[Coord<2>(3, 3)].testValue, 64.0);
        TS_ASSERT_EQUALS(writer->getGrid( 8)[Coord<2>(4, 3)].testValue, 65.0);

        TS_ASSERT_EQUALS(writer->getGrid(10)[Coord<2>(3, 3)].testValue, 64.0 + 1234);
        TS_ASSERT_EQUALS(writer->getGrid(10)[Coord<2>(4, 3)].testValue, 65.0 + 1234);
    }

    void testInvalidHandler()
    {
        // boost::shared_ptr<SteeringInteractor> interactor;

        if (mpiLayer.rank() == 0) {
            steerer->addAction(new CommandServer<TestCell<2> >::PassThroughAction("nonExistentHandler", "blah"));
            StringVec res;

            // interactor.reset(new SteeringInteractor);
            // Wrapper<SteeringInteractor> wrapper(interactor);
            // boost::thread interactorThread(wrapper);
            // interactor->waitForStartup();
            // std::cout << "waiting for join\n";
            // interactorThread.join();
            // std::cout << "party on, wayne!\n";

            // res = steerer->sendCommandWithFeedback("nonExistentAction  1 2 3", 1);
            // TS_ASSERT_EQUALS(res.size(), 1);
            // TS_ASSERT_EQUALS(res[0], "command not found: nonExistentAction\n");
            // std::cout << "sending nonexistendhandler\n";
            // res = steerer->sendCommandWithFeedback("nonExistentHandler 1 2 3", 1);
            // std::cout << "blah\n";
            // TS_ASSERT_EQUALS(res.size(), 1);
            // TS_ASSERT_EQUALS(res[0], "handler not found: nonExistentHandler\n");
        }

        // std::cout << "sleeping\n";
        // sleep(10);
        // sim->run();
    }

    void testGetSet()
    {
        // std::cout << "sleeping\n";
        // sleep(10);
        // sim->run();
    }


private:
    MPILayer mpiLayer;
    boost::shared_ptr<StripingSimulator<TestCell<2> > > sim;
    RemoteSteerer<TestCell<2> > *steerer;
    ParallelMemoryWriter<TestCell<2> > *writer;
    // fixme: test set/get
    // fixme: test remotesteerer with 1 proc
    // fixme: add help function
    // fixme: move stringvec to dedicated class
    // fixme: test unknown handler
    // fixme: fix stuff in parallememorywriter
    // fixme: refactor steerer interface
    // fixme: fix cars example
    // fixme: fix gameoflive_life example
    // fixme: fix communicator handling in mpilayer
};

}
