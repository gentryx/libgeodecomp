#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/remotesteerer/interactor.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

class InteractorTest : public CxxTest::TestSuite
{
public:
    void testThreaded()
    {
        int port = 47111;
        StringVec expectedFeedback;
        expectedFeedback << "bingo bongo\n";

        // listen on port "port"
        boost::asio::io_service ioService;
        tcp::acceptor acceptor(ioService, tcp::endpoint(tcp::v4(), port));
        tcp::socket socket(ioService);

        // start the interactor and wait until it has sent its commands
        Interactor<int> interactor("command blah\n", 1, true, port);
        interactor.waitForStartup();

        // grab the data from the interactor:
        boost::system::error_code errorCode;
        acceptor.accept(socket, errorCode);
        boost::asio::streambuf buf;
        size_t length = boost::asio::read_until(socket, buf, '\n', errorCode);
        // gah, converting streambuf to string couldn't be any easier...
        std::istream lineBuf(&buf);
        std::string line(length, 'X');
        lineBuf.read(&line[0], length);

        // write back some feedpack
        boost::asio::write(
            socket,
            boost::asio::buffer(expectedFeedback[0]),
            boost::asio::transfer_all(),
            errorCode);

        // check the resupts
        StringVec tokens = StringOps::tokenize(line, " \r\n");
        StringVec expected;
        expected << "command"
                 << "blah";
        TS_ASSERT_EQUALS(tokens, expected);
        interactor.waitForCompletion();
        TS_ASSERT_EQUALS(interactor.feedback(), expectedFeedback);
    }
};

}

}
