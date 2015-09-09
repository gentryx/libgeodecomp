#ifndef LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H
#define LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14
#include <utility>
#endif

#include <hpx/lcos/broadcast.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <libgeodecomp/communication/hpxcomponentregsitrationhelper.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

template <typename CARGO, typename BUFFER=hpx::lcos::local::receive_buffer<CARGO> >
class HPXReceiver : public hpx::components::simple_component_base<HPXReceiver<CARGO> >
{
public:
    typedef CARGO Cargo;
    typedef BUFFER Buffer;

    static hpx::future<boost::shared_ptr<HPXReceiver> > make(const std::string& name, std::size_t rank = 0)
    {
        HPXComponentRegistrator<HPXReceiver> thisEnsuresHPXRegistrationCodeIsRunPriorToComponentCreation;

        hpx::id_type id = hpx::new_<HPXReceiver>(hpx::find_here()).get();
        hpx::register_with_basename(name, id, rank).get();
        return hpx::get_ptr<HPXReceiver>(id);
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

        return std::move(ids);
    }

    static std::vector<CARGO> allGather(
        const CARGO& data,
        std::size_t rank,
        std::size_t size,
        const std::string& name)
    {
        auto receiver = HPXReceiver<CARGO>::make(name, rank).get();
        std::vector<hpx::future<hpx::id_type> > futures = HPXReceiver<CARGO>::find_all(name, size);
        std::vector<hpx::id_type> ids = hpx::util::unwrapped(std::move(futures));
        std::vector<CARGO> vec;
        vec.reserve(size);

        hpx::lcos::broadcast_apply<typename HPXReceiver::receiveAction>(ids, rank, data);

        for (std::size_t i = 0; i < size; ++i) {
            vec << receiver->get(i).get();
        }

        return std::move(vec);
    }

    virtual ~HPXReceiver()
    {}

    void receive(std::size_t step, Cargo&& val)
    {
        buffer.store_received(step, std::forward<Cargo>(val));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXReceiver, receive, receiveAction);

    hpx::future<Cargo> get(std::size_t step)
    {
        return buffer.receive(step);
    }

private:
    Buffer buffer;

    LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANTIATIONS(HPXReceiver);
};

}

LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE(typename CARGO, LibGeoDecomp::HPXReceiver<CARGO>)

#endif
