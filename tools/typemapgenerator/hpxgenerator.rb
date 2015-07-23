load 'basicgenerator.rb'

# Here we generate the C++ code that will in turn create the typemaps
# for HPX Serialization.
class HPXGenerator
  include BasicGenerator

  def initialize(template_path="./", namespace=nil, macro_guard_hpx=nil)
    @serialization_class_name = "HPXSerialization"
    @serialization_namespace = "hpx"
    init_generator(template_path, namespace)
    @macro_guard = macro_guard_hpx
  end

  def base_object_name
    "hpx::serialization::base_object"
  end

  def class_registrations(options)
    ret = []

    options.classes.each do |klass|
      if options.template_parameters[klass].size == 0
        # if options.is_abstract[klass]
        ret.push "HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(#{klass});"
        # else
        #   ret.push "HPX_SERIALIZATION_REGISTER_CLASS(#{klass});"
        # end
      else
        params1 = render_template_params1(options.template_parameters[klass])
        params2 = render_template_params2(options.template_parameters[klass])
        ret.push "HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <#{params1}>), (#{klass}#{params2}));"
        if !options.is_abstract[klass]
          ret.push "HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE((template <#{params1}>), (#{klass}#{params2}));"
        # else
        #   ret.push "HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(#{klass});"
        end
      end
    end

    return ret.join("\n") + "\n\n"
  end

  def class_registrations_source(options)
    ret = []

    options.classes.each do |klass|
      if options.template_parameters[klass].size == 0
        if !options.is_abstract[klass]
          ret.push "HPX_SERIALIZATION_REGISTER_CLASS(#{klass})"
        end
      end
    end

    return ret.join("\n") + "\n\n"
  end

  # wraps the code generation for multiple typemaps.
  def generate_forest(options)
    return [generate_header(options),
            generate_source(options)]
  end
end
