#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/storage/dataaccessor.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO

#include <libgeodecomp/io/remotesteerer.h>
#include <libgeodecomp/io/remotesteerer/interactor.h>

#endif

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

DEFINE_DATAACCESSOR(TestValueAccessor, TestCell<2>, double, testValue);

class RemoteSteererTest : public CxxTest::TestSuite
{
public:
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
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
                TestCell<2> cell = grid->get(*i);
                cell.testValue += value;
                grid->set(*i, cell);
            }

            return true;
        }

    };

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
#endif

    void setUp()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
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
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        sim.reset();
#endif
    }

    void testBasic()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new FlushAction);
            StringVec feedback = steerer->sendCommandWithFeedback("flush 1234 9", 1);
            TS_ASSERT_EQUALS(std::size_t(1), feedback.size());
            TS_ASSERT_EQUALS("flush received", feedback[0]);
        }

        sim->run();

        TS_ASSERT_EQUALS(writer->getGrids().size(), static_cast<std::size_t>(16));
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
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        if (mpiLayer.rank() == 0) {
            StringVec res;
            res = steerer->sendCommandWithFeedback("nonExistentAction  1 2 3", 2);

            std::cout << "res: " << res << "\n";
            TS_ASSERT_EQUALS(res.size(), std::size_t(2));
            TS_ASSERT_EQUALS(res[0], "command not found: nonExistentAction");
            TS_ASSERT_EQUALS(res[1], "try \"help\"");
        }
#endif
    }

    void testInvalidHandler()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 0) {
            steerer->addAction(new PassThroughAction<TestCell<2> >("nonExistentHandler", "blah"));
            interactor.reset(new Interactor("nonExistentHandler bongo", 2, true, port));
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequestsQueue().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();

        sim->run();

        if (mpiLayer.rank() == 0) {
            interactor->waitForCompletion();
            StringVec expected;
            expected << "handler not found: nonExistentHandler"
                     << "handler not found: nonExistentHandler";

            TS_ASSERT_EQUALS(interactor->feedback(), expected);
        }
#endif
    }

    void testHandlerNotFound1()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new PassThroughAction<TestCell<2> >("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 1) {
            steerer->addHandler(new EchoHandler());
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequestsQueue().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(feedback.size(), std::size_t(2));
            TS_ASSERT_EQUALS(feedback[0], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[1], "echo reply from rank 1 at time step 0 with cargo »romeo,tango,yankee,papa,echo«");
        }
#endif
    }

    void testHandlerNotFound2()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new PassThroughAction<TestCell<2> >("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 0) {
            steerer->addHandler(new EchoHandler());
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequestsQueue().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(feedback.size(), std::size_t(2));
            TS_ASSERT_EQUALS(feedback[1], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[0], "echo reply from rank 0 at time step 0 with cargo »romeo,tango,yankee,papa,echo«");
        }
#endif
    }

    void testHandlerNotFound3()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        if (mpiLayer.rank() == 0) {
            steerer->addAction(new PassThroughAction<TestCell<2> >("echo", "blah"));
        }
        mpiLayer.barrier();
        boost::shared_ptr<Interactor> interactor;

        if (mpiLayer.rank() == 1) {
            interactor.reset(new Interactor("echo romeo,tango,yankee,papa,echo", 2, true, port));
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequestsQueue().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();
        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            StringVec feedback = interactor->feedback();
            TS_ASSERT_EQUALS(feedback.size(), std::size_t(2));
            TS_ASSERT_EQUALS(feedback[0], "handler not found: echo");
            TS_ASSERT_EQUALS(feedback[1], "handler not found: echo");
        }
#endif
    }

    void testGetSet()
    {
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
        steerer->addDataAccessor(new TestValueAccessor());
        boost::shared_ptr<Interactor> interactor;
        mpiLayer.barrier();

        if (mpiLayer.rank() == 0) {
            interactor.reset(new Interactor("get_testValue 2 1 3", 2, true, port));
        }

        // sleep until the request has made it into the pipeline
        if (mpiLayer.rank() == 0) {
            while (steerer->pipe->copySteeringRequestsQueue().size() == 0) {
                usleep(10000);
            }
        }
        mpiLayer.barrier();

        sim->run();

        if (interactor) {
            interactor->waitForCompletion();
            std::cout << "feedback: " << interactor->feedback();
        }
#endif
    }

private:
#if defined LIBGEODECOMP_WITH_THREADS && defined LIBGEODECOMP_WITH_BOOST_ASIO
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
