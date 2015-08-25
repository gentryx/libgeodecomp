#ifndef LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H
#define LIBGEODECOMP_COMMUNICATION_HPXRECEIVER_H

#include <hpx/lcos/local/receive_buffer.hpp>
#include <hpx/runtime/get_ptr.hpp>

namespace LibGeoDecomp {

template <typename CARGO, typename BUFFER=hpx::lcos::local::receive_buffer<CARGO> >
class HPXReceiver : public hpx::components::simple_component_base<HPXReceiver<CARGO> >
{
public:
    typedef CARGO Cargo;
    typedef BUFFER Buffer;

    static hpx::future<boost::shared_ptr<HPXReceiver> > make(const std::string& name)
    {
        hpx::id_type id = hpx::new_<HPXReceiver>(hpx::find_here()).get();
        hpx::register_id_with_basename(name.c_str(), id, 0).get();
        return hpx::get_ptr<HPXReceiver>(id);
    }

    static hpx::future<hpx::id_type> find(const std::string& name)
    {
        std::vector<hpx::future<hpx::id_type> > ids = hpx::find_all_ids_from_basename(name.c_str(), 1);
        if (ids.size() != 1) {
            throw std::logic_error("Unexpected amount of PatchProviders found in AGAS, expected exactly 1");
        }

        return std::move(ids[0]);
    }

    void receive(std::size_t step, Cargo&& val)
    {
        buffer.store_received(step, std::move(val));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXReceiver, receive, receiveAction);

    hpx::future<Cargo> get(std::size_t step)
    {
        return buffer.receive(step);
    }

private:
    Buffer buffer;
};

}
#endif
