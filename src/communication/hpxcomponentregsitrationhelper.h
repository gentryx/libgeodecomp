#ifndef LIBGEODECOMP_COMMUNICATION_HPXCOMPONENTREGSITRATIONHELPER_H
#define LIBGEODECOMP_COMMUNICATION_HPXCOMPONENTREGSITRATIONHELPER_H

namespace {
template<typename COMPONENT>
class hpx_plugin_exporter_factory;

template<typename COMPONENT>
class init_registry_factory_static;

template<typename T>
class hpx_plugin_exporter_registry;

}

// fixme: lacks deletion of parentheses
#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE(PARAMS, TEMPLATE)  \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any> * hpx_exported_plugins_list_hpx_factory(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename COMPONENT>                                        \
    class hpx_plugin_exporter_factory;                                  \
                                                                        \
    template<PARAMS>                                                    \
    class hpx_plugin_exporter_factory<TEMPLATE > \
    {                                                                   \
    public:                                                             \
        hpx_plugin_exporter_factory()                                   \
        {                                                               \
            static hpx::util::plugin::concrete_factory< hpx::components::component_factory_base, hpx::components::component_factory<hpx::components::simple_component<TEMPLATE>> > cf; \
            hpx::util::plugin::abstract_factory<hpx::components::component_factory_base>* w = &cf; \
                                                                        \
            std::string actname(typeid(hpx::components::simple_component<TEMPLATE>).name()); \
            boost::algorithm::to_lower(actname);                        \
            hpx_exported_plugins_list_hpx_factory()->insert( std::make_pair(actname, w)); \
        }                                                               \
                                                                        \
        static hpx_plugin_exporter_factory instance;                    \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    hpx_plugin_exporter_factory<TEMPLATE> hpx_plugin_exporter_factory<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any>* hpx_exported_plugins_list_hpx_factory(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename COMPONENT>                                        \
    class init_registry_factory_static;                                 \
                                                                        \
    template<PARAMS>                                                    \
    class init_registry_factory_static<TEMPLATE >                       \
    {                                                                   \
    public:                                                             \
        init_registry_factory_static<TEMPLATE>()                        \
        {                                                               \
            hpx::components::static_factory_load_data_type data = { typeid(hpx::components::simple_component<TEMPLATE>).name(), hpx_exported_plugins_list_hpx_factory }; \
            hpx::components::init_registry_factory(data);               \
        }                                                               \
                                                                        \
        static init_registry_factory_static<TEMPLATE> instance;         \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    init_registry_factory_static<TEMPLATE> init_registry_factory_static<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace components {                                              \
                                                                        \
    template <PARAMS> struct unique_component_name<hpx::components::component_factory<hpx::components::simple_component<TEMPLATE > > > \
    {                                                                   \
        typedef char const* type;                                       \
                                                                        \
        static type call(void)                                          \
        {                                                               \
            return typeid(hpx::components::simple_component<TEMPLATE >).name(); \
        }                                                               \
    };                                                                  \
                                                                        \
    }                                                                   \
    }                                                                   \
                                                                        \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any> * hpx_exported_plugins_list_hpx_registry(); \
                                                                        \
    namespace {                                                         \
                                                                        \
    template<typename T>                                                \
    class hpx_plugin_exporter_registry;                                 \
                                                                        \
    template<PARAMS>                                                    \
    class hpx_plugin_exporter_registry<TEMPLATE> \
    {                                                                   \
    public:                                                             \
        hpx_plugin_exporter_registry()                                  \
        {                                                               \
            static hpx::util::plugin::concrete_factory< hpx::components::component_registry_base, hpx::components::component_registry<hpx::components::simple_component<TEMPLATE>, ::hpx::components::factory_check> > cf; \
            hpx::util::plugin::abstract_factory<hpx::components::component_registry_base>* w = &cf; \
            std::string actname(typeid(hpx::components::simple_component<TEMPLATE>).name()); \
            boost::algorithm::to_lower(actname);                        \
            hpx_exported_plugins_list_hpx_registry()->insert( std::make_pair(actname, w)); \
        }                                                               \
                                                                        \
        static hpx_plugin_exporter_registry instance;                   \
    };                                                                  \
                                                                        \
    template<PARAMS>                                                    \
    hpx_plugin_exporter_registry<TEMPLATE> hpx_plugin_exporter_registry<TEMPLATE>::instance; \
                                                                        \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace components {                                              \
                                                                        \
    template <PARAMS>                                                   \
    struct unique_component_name<hpx::components::component_registry<hpx::components::simple_component<TEMPLATE >, ::hpx::components::factory_check> > \
    {                                                                   \
        typedef char const* type;                                       \
        static type call (void)                                         \
        {                                                               \
            return typeid(hpx::components::simple_component<TEMPLATE >).name(); \
        }                                                               \
    };                                                                  \
                                                                        \
    }                                                                   \
    }                                                                   \
                                                                        \
    namespace hpx {                                                     \
    namespace traits {                                                  \
                                                                        \
    template<PARAMS, typename ENABLE>                                   \
    __attribute__((visibility("default")))                              \
    components::component_type component_type_database<CARGO, ENABLE>::get() \
    {                                                                   \
        return value;                                                   \
    }                                                                   \
                                                                        \
    template<PARAMS, typename ENABLE>                                   \
    __attribute__((visibility("default")))                              \
    void component_type_database<CARGO, ENABLE>::set( components::component_type t) \
    {                                                                   \
        value = t;                                                      \
    }                                                                   \
                                                                        \
    }                                                                   \
    };

// fixme: lacks deletion of parentheses
#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANCE(TYPENAME) \
    template struct hpx::components::component_factory<hpx::components::simple_component<TYPENAME>>; \
    template struct hpx::components::component_registry< hpx::components::simple_component<TYPENAME>, ::hpx::components::factory_check>;

#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE_INSTANTIATIONS(TYPENAME) \
    virtual hpx_plugin_exporter_factory<TYPENAME> hpx_plugin_exporter_factory_registration() \
    {                                                                   \
        return hpx_plugin_exporter_factory<TYPENAME>::instance;         \
    }                                                                   \
                                                                        \
    virtual init_registry_factory_static<TYPENAME> hpx_init_registry_factory_static_registration() \
    {                                                                   \
        return init_registry_factory_static<TYPENAME>::instance;        \
    }                                                                   \
                                                                        \
    virtual hpx_plugin_exporter_registry<TYPENAME> hpx_plugin_exporter_registry_registration() \
    {                                                                   \
        return hpx_plugin_exporter_registry<TYPENAME>::instance;        \
    }

#endif
