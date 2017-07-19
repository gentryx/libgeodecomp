#ifndef LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H
#define LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H

#include <libgeodecomp/config.h>
#include <hpx/config.hpp>
#ifdef LIBGEODECOMP_WITH_CPP14
#include <utility>
#endif

#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/stringops.h>

#define LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(CARGO)                      \
    typedef LibGeoDecomp::HPXReceiver<CARGO>::receiveAction DummyReceiver_ ## CARGO ## _ReceiveAction; \
    HPX_REGISTER_ACTION_DECLARATION(DummyReceiver_ ## CARGO ## _ReceiveAction); \
    HPX_REGISTER_ACTION(DummyReceiver_ ## CARGO ## _ReceiveAction);     \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(DummyReceiver_ ## CARGO ## _ReceiveAction); \
    HPX_REGISTER_BROADCAST_APPLY_ACTION(DummyReceiver_ ## CARGO ## _ReceiveAction); \
    typedef hpx::components::managed_component<LibGeoDecomp::HPXReceiver<CARGO> > receiver_type_ ## CARGO; \
    HPX_REGISTER_COMPONENT(receiver_type_ ## CARGO , DummyReceiver_ ## CARGO); \
                                                                        \
    typedef LibGeoDecomp::HPXReceiver<std::vector<CARGO> >::receiveAction DummyReceiver_vector_ ## CARGO ## _ReceiveAction; \
    HPX_REGISTER_ACTION_DECLARATION(DummyReceiver_vector_ ## CARGO ## _ReceiveAction); \
    HPX_REGISTER_ACTION(DummyReceiver_vector_ ## CARGO ## _ReceiveAction);     \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(DummyReceiver_vector_ ## CARGO ## _ReceiveAction); \
    HPX_REGISTER_BROADCAST_APPLY_ACTION(DummyReceiver_vector_ ## CARGO ## _ReceiveAction); \
    typedef hpx::components::managed_component<LibGeoDecomp::HPXReceiver<std::vector<CARGO>> > receiver_type_vector_ ## CARGO; \
    HPX_REGISTER_COMPONENT(receiver_type_vector_ ## CARGO , DummyReceiver_vector_ ## CARGO);

namespace LibGeoDecomp {

template <typename CARGO, typename BUFFER=hpx::lcos::local::receive_buffer<CARGO> >
class HPX_ALWAYS_EXPORT HPXReceiver : public hpx::components::managed_component_base<HPXReceiver<CARGO> >
{
public:
    typedef CARGO Cargo;
    typedef BUFFER Buffer;

    static hpx::future<std::shared_ptr<HPXReceiver> > make(const std::string& name, std::size_t rank = 0)
    {
        return hpx::new_<HPXReceiver>(hpx::find_here()).then(
            [name, rank](hpx::future<hpx::id_type> idFuture)
            {
                hpx::id_type id = idFuture.get();
                hpx::future<bool> f = hpx::register_with_basename(name, id, rank);
                return f.then(
                    [id](hpx::future<bool>)
                    {
                        return hpx::get_ptr<HPXReceiver>(id);
                    });
            });
    }

    static hpx::future<hpx::id_type> find(const std::string& name)
    {
        std::vector<hpx::future<hpx::id_type> > ids = hpx::find_all_from_basename(name, 1);
        if (ids.size() != 1) {
            throw std::logic_error("Unexpected amount of HPXReceivers found in AGAS, expected exactly 1");
        }

        return std::move(ids[0]);
    }

    static std::vector<hpx::future<hpx::id_type> > find_all(const std::string& name, std::size_t num)
    {
        std::vector<hpx::future<hpx::id_type> > ids = hpx::find_all_from_basename(name, num);
        if (ids.size() != num) {
            throw std::logic_error("Unexpected amount of HPXReceivers found in AGAS, exected exactly " +
                                   StringOps::itoa(num));
        }

        return ids;
    }

    virtual ~HPXReceiver()
    {}

    void receive(std::size_t step, Cargo val)
    {
        buffer.store_received(step, std::move(val));
    }
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(HPXReceiver, receive, receiveAction);

    hpx::future<Cargo> get(std::size_t step)
    {
        return buffer.receive(step);
    }

    static std::vector<CARGO> allGather(
        const CARGO& data,
        std::size_t rank,
        std::size_t size,
        const std::string& name)
    {
        hpx::future<std::shared_ptr<HPXReceiver> > receiverFuture = hpx::dataflow(
            [rank, data](
                hpx::future<std::shared_ptr<HPXReceiver> > receiverFuture,
                std::vector<hpx::future<hpx::id_type> > idsFuture
            )
            {
                hpx::lcos::broadcast_apply<typename HPXReceiver::receiveAction>(
                    hpx::util::unwrapped(idsFuture), rank, data);
                return receiverFuture.get();
            },
            HPXReceiver<CARGO>::make(name, rank),
            HPXReceiver<CARGO>::find_all(name, size)
        );

        std::vector<CARGO> vec;
        vec.reserve(size);

        auto receiver = receiverFuture.get();

        for (std::size_t i = 0; i < size; ++i) {
            vec << receiver->get(i).get();
        }

        return std::move(vec);
    }

private:
    Buffer buffer;
};

}

#endif
