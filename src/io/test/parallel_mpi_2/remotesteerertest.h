#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#ifdef LIBGEODECOMP_FEATURE_THREADS
#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/remotesteerer/interactor.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#endif

using namespace LibGeoDecomp;

namespace LibGeoDecomp {


class RemoteSteererTest : public CxxTest::TestSuite
{
public:
#ifdef LIBGEODECOMP_FEATURE_THREADS
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
#endif

    class EchoHandler : public Handler<TestCell<2> >
    {
    public:
        EchoHandler() :
           Handler<TestCell<2> >("echo")
        {}

        virtual bool operator()(const StringVec& parameters, Pipe& pipe, GridType *grid, const Region<Topology::DIM>& validRegion, unsigned step)
        {
            std::stringstream buf;
            buf << "echo reply from rank " << MPILayer().rank()
                << " at time step " << step
                << " with cargo »" << parameters[0] << "«\n";
            pipe.addSteeringFeedback(buf.str());
            return true;
        }
    };

    void setUp()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        unsigned steererPeriod = 3;
        unsigned writerPeriod = 2;
        port = 47112;
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
#endif
    }

    void tearDown()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        sim.reset();
#endif
    }

    void testBasic()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new FlushAction);
            StringVec feedback = steerer->sendCommandWithFeedback("flush 1234 9", 1);
            TS_ASSERT_EQUALS(1, feedback.size());
            TS_ASSERT_EQUALS("flush received", feedback[0]);
        }

        sim->run();

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
#endif
    }

    void testNonExistentAction()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        if (mpiLayer.rank() == 0) {
            StringVec res;
            res = steerer->sendCommandWithFeedback("nonExistentAction  1 2 3", 1);

            TS_ASSERT_EQUALS(res.size(), 1);
            TS_ASSERT_EQUALS(res[0], "command not found: nonExistentAction");
        }
#endif
    }

    void testInvalidHandler()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 0) {
            steerer->addAction(new CommandServer<TestCell<2> >::PassThroughAction("nonExistentHandler", "blah"));
            interactor.reset(new Interactor("nonExistentHandler bongo\nwait\n", 1, true, port));
            interactor->waitForStartup();
        }

        sim->run();

        if (mpiLayer.rank() == 0) {
            interactor->waitForCompletion();
            StringVec expected;
            expected << "handler not found: nonExistentHandler";

            TS_ASSERT_EQUALS(interactor->feedback(), expected);
        }
#endif
    }

    template<typename CELL_TYPE>
    class GetHandler : public Handler<CELL_TYPE>
    {
    public:
        typedef typename CELL_TYPE::Topology Topology;
        typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
        static const int DIM = Topology::DIM;

        GetHandler() :
            Handler<CELL_TYPE>("get")
        {}

        virtual bool operator()(const StringVec& parameters, Pipe& pipe, GridType *grid, const Region<DIM>& validRegion, unsigned step)
        {
            Coord<DIM> c;
            int index;

            for (index = 0; index < DIM; ++index) {
                c[index] = StringOps::atoi(parameters[index]);
            }

            const std::string& member = parameters[index];

            std::cout << "get " << c << " member: " << member << "\n";
            pipe.addSteeringFeedback("bingo bongo");

            return true;
        }
    };

    void testHandlerNotFound1()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new CommandServer<TestCell<2> >::PassThroughAction("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 1) {
            steerer->addHandler(new EchoHandler());
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
            interactor->waitForStartup();
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequests().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(2, feedback.size());
            TS_ASSERT_EQUALS(feedback[0], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[1], "echo reply from rank 1 at time step 0 with cargo »romeo,tango,yankee,papa,echo«");
        }
#endif
    }

    void testHandlerNotFound2()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new CommandServer<TestCell<2> >::PassThroughAction("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 0) {
            steerer->addHandler(new EchoHandler());
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
            interactor->waitForStartup();
        }

        // sleep until the request has made it into the pipeline
        usleep(250000);
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(2, feedback.size());
            TS_ASSERT_EQUALS(feedback[1], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[0], "echo reply from rank 0 at time step 0 with cargo »romeo,tango,yankee,papa,echo«");
        }
#endif
    }

    void testHandlerNotFound3()
    {
#ifdef LIBGEODECOMP_FEATURE_THREADS
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new CommandServer<TestCell<2> >::PassThroughAction("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 1) {
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
            interactor->waitForStartup();
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequests().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(2, feedback.size());
            TS_ASSERT_EQUALS(feedback[0], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[1], "handler not found: echo");
        }
#endif
    }

//     void testGetSet()
//     {
// #ifdef LIBGEODECOMP_FEATURE_THREADS
//         mpiLayer.barrier();
//         std::cout << "------------------------------------------------------------------------\n";
//         boost::shared_ptr<Interactor> interactor;
//         std::cout << "meatboy1--------------------------------------\n";
//         if (mpiLayer.rank() == 0) {
//             steerer->addHandler(new GetHandler<TestCell<2> >());
//             interactor.reset(new Interactor("get 1 2 masupilami", 2, true, port));
//         }
//         std::cout << "meatboy2--------------------------------------\n";
//         if (interactor) {
//             interactor->waitForStartup();
//         }
//         std::cout << "meatboy3--------------------------------------\n";
//         sim->run();
//         std::cout << "meatboy4--------------------------------------\n";
//         if (interactor) {
//             interactor->waitForCompletion();
//             std::cout << "feedback: " << interactor->feedback();
//         }
//         std::cout << "meatboy5--------------------------------------\n";
// #endif
//     }

private:
#ifdef LIBGEODECOMP_FEATURE_THREADS
    MPILayer mpiLayer;
    boost::shared_ptr<StripingSimulator<TestCell<2> > > sim;
    RemoteSteerer<TestCell<2> > *steerer;
    ParallelMemoryWriter<TestCell<2> > *writer;
    int port;
    // fixme: test set/get
    // fixme: test remotesteerer with 1 proc
    // fixme: add help function
    // fixme: refactor steerer interface
    // fixme: fix cars example
    // fixme: fix gameoflive_life example
    // fixme: fix communicator handling in mpilayer
    // fixme: test requeueing of unhandled requests
    // fixme: add rank to logging output
#endif
};

}
