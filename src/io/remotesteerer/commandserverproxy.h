#ifndef LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVERPROXY_H
#define LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVERPROXY_H

#include <libgeodecomp/io/remotesteerer/commandserver.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

/**
 * This class tunnels communication of Steerer instances to a remote
 * CommandServer. The root will collect all messages and let its local
 * CommandServer process those.
 *
 * fixme: needs test
 */
class CommandServerProxy
{
public:
    CommandServerProxy(MPI::Intracomm& comm, CommandServer *commandServer) :
        comm(comm),
        commandServer(commandServer)
    {}

    void sendMessage(std::string msg)
    {
        if (comm.Get_rank() == 0) {
            commandServer->sendMessage("thread 0: " + msg);
        } else {
            msgBuffer.push_back(msg);
        }
    }

    void collectMessages()
    {
        int vectorSize = 0;
        int stringSize = 0;
        int msgCount[comm.Get_size()];
        if (comm.Get_rank() == 0) {
            comm.Gather(&vectorSize, 1, MPI::INT, msgCount, 1, MPI::INT, 0);

            // fixme: early return/continue
            for (int i = 1; i < comm.Get_size(); ++i) {
                if (msgCount[i] > 0) {
                    for (int j = 0; j < msgCount[i]; ++j) {
                        comm.Recv(&stringSize, 1, MPI::INT, i, 1);
                        char charBuffer[stringSize + 1];
                        charBuffer[stringSize] = '\0';
                        comm.Recv(charBuffer, stringSize, MPI::CHAR, i, 2);
                        std::string thread = "thread ";
                        std::stringstream ss;
                        ss << i;
                        thread += ss.str();
                        thread += ": ";
                        commandServer->sendMessage(thread + std::string(charBuffer));
                    }
                }
            }
        } else {
            // fixme: decompose
            vectorSize = msgBuffer.size();
            comm.Gather(&vectorSize, 1, MPI::INT, msgCount, 1, MPI::INT, 0);

            if (vectorSize > 0) {
                for (int i = 0; i < vectorSize; ++i) {
                    stringSize = msgBuffer.at(i).size();
                    comm.Send(&stringSize, 1, MPI::INT, 0, 1);
                    char* tmp = const_cast<char*>(msgBuffer.at(i).c_str());
                    comm.Send(tmp, stringSize, MPI::CHAR, 0, 2);
                }
                msgBuffer.clear();
            }
        }
    }

private:
    std::vector<std::string> msgBuffer;
    MPI::Intracomm& comm;
    CommandServer *commandServer;
};

}

}

#endif
