#ifndef LIBGEODECOMP_COMMUNICATION_HPXCOMPONENTREGSITRATIONHELPER_H
#define LIBGEODECOMP_COMMUNICATION_HPXCOMPONENTREGSITRATIONHELPER_H

namespace {

/**
 * Internal helper class
 */
template<typename COMPONENT>
class hpx_plugin_exporter_factory;

/**
 * Internal helper class
 */
template<typename COMPONENT>
class init_registry_factory_static;

}

namespace LibGeoDecomp {

/**
 * Instantiate this template to ensure the instantiation of an HPX
 * component template is actually registered. See HPXReceiver for an
 * example on how to use this class and its assorted macros.
 */
template<typename COMPONENT>
class HPXComponentRegistrator
{
public:
    virtual ~HPXComponentRegistrator()
    {}

    static hpx::components::component_factory<hpx::components::simple_component<COMPONENT> > *instanceA;
    static hpx::components::component_registry< hpx::components::simple_component<COMPONENT>, ::hpx::components::factory_check> *instanceB;

    virtual hpx::components::component_factory<hpx::components::simple_component<COMPONENT> > *foo1()
    {
        instanceA = new hpx::components::component_factory<hpx::components::simple_component<COMPONENT> >(0, 0, false);
        return instanceA;
    }

    virtual hpx::components::component_registry< hpx::components::simple_component<COMPONENT>, ::hpx::components::factory_check> *foo2()
    {
        instanceB = new hpx::components::component_registry< hpx::components::simple_component<COMPONENT>, ::hpx::components::factory_check>();
        return instanceB;
    }
};

template<typename COMPONENT>
hpx::components::component_factory<hpx::components::simple_component<COMPONENT> > *HPXComponentRegistrator<COMPONENT>::instanceA;

template<typename COMPONENT>
hpx::components::component_registry< hpx::components::simple_component<COMPONENT>, ::hpx::components::factory_check> *HPXComponentRegistrator<COMPONENT>::instanceB;

}

#define LIBGEODECOMP_HPX_PLUGIN_REGISTRY                                \
    BOOST_PP_CAT(                                                       \
        BOOST_PP_CAT(                                                   \
            HPX_MANGLE_NAME(hpx)                                        \
          , BOOST_PP_CAT(                                               \
                _exported_plugins_list_                                 \
              , HPX_MANGLE_NAME(hpx)                                    \
            )                                                           \
        )                                                               \
      , _registry                                                       \
    )                                                                   \
/**/

#define LIBGEODECOMP_HPX_PLUGIN_FACTORY                                 \
    BOOST_PP_CAT(                                                       \
        BOOST_PP_CAT(                                                   \
            HPX_MANGLE_NAME(hpx)                                        \
          , BOOST_PP_CAT(                                               \
                _exported_plugins_list_                                 \
              , HPX_MANGLE_NAME(hpx)                                    \
            )                                                           \
        )                                                               \
      , _factory                                                        \
    )                                                                   \

// fixme: lacks deletion of parentheses
#define LIBGEODECOMP_REGISTER_HPX_COMPONENT_TEMPLATE(PARAMS, TEMPLATE)  \
    extern "C" __attribute__((visibility ("default")))                  \
    std::map<std::string, boost::any>* LIBGEODECOMP_HPX_PLUGIN_FACTORY(); \
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
            LIBGEODECOMP_HPX_PLUGIN_FACTORY()->insert( std::make_pair(actname, w)); \
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
            hpx::components::static_factory_load_data_type data = {     \
                typeid(hpx::components::simple_component<TEMPLATE>).name(), \
                LIBGEODECOMP_HPX_PLUGIN_FACTORY                         \
            };                                                          \
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
    std::map<std::string, boost::any>* LIBGEODECOMP_HPX_PLUGIN_REGISTRY(); \
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
            LIBGEODECOMP_HPX_PLUGIN_REGISTRY()->insert( std::make_pair(actname, w)); \
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
