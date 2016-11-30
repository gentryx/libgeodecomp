#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/remotesteerer/interactor.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

class InteractorTest : public CxxTest::TestSuite
{
public:
    void testSerial()
    {
        // fixme

        // MPILayer mpiLayer;
        // int port = 47113;
        // StringVec expectedFeedback;
        // expectedFeedback << "bingo bongo";

        // if (mpiLayer.rank() == 0) {
        //     // listen on port "port"
        //     boost::asio::io_service ioService;
        //     tcp::acceptor acceptor(ioService, tcp::endpoint(tcp::v4(), port));
        //     tcp::socket socket(ioService);

        //     mpiLayer.barrier();

        //     // grab the data from the interactor:
        //     boost::system::error_code errorCode;
        //     acceptor.accept(socket, errorCode);
        //     boost::asio::streambuf buf;
        //     std::size_t length = boost::asio::read_until(socket, buf, '\n', errorCode);
        //     // gah, converting streambuf to string couldn't be any easier...
        //     std::istream lineBuf(&buf);
        //     std::string line(length, 'X');
        //     lineBuf.read(&line[0], length);

        //     // write back some feedback
        //     boost::asio::write(
        //         socket,
        //         boost::asio::buffer("bingo bongo\n"),
        //         boost::asio::transfer_all(),
        //         errorCode);

        //     // check the results
        //     StringVec tokens = StringOps::tokenize(line, " \r\n");
        //     StringVec expected;
        //     expected << "command"
        //              << "blah";
        //     TS_ASSERT_EQUALS(tokens, expected);
        // } else {
        //     mpiLayer.barrier();

        //     // start the interactor and wait until it has sent its commands
        //     Interactor interactor("command blah\n", 1, true, port);

        //     // check the results
        //     interactor.waitForCompletion();

        //     StringVec actualFeedback = interactor.feedback();
        //     TS_ASSERT_EQUALS(actualFeedback, expectedFeedback);
        // }
    }
};

}

}
