load 'basicgenerator.rb'

# Here we generate the C++ code that will in turn create the typemaps
# for Boost Serialization.
class BoostGenerator
  include BasicGenerator

  def initialize(template_path="./", namespace=nil, macro_guard_boost=nil)
    @serialization_class_name = "BoostSerialization"
    @serialization_namespace = "boost"
    init_generator(template_path, namespace)
    @macro_guard = macro_guard_boost
  end

  def base_object_name
    "boost::serialization::base_object"
  end

  # wraps the code generation for multiple typemaps.
  def generate_forest(options)
    return [generate_header(options)]
  end
end
