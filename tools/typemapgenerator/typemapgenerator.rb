require 'mpiparser'
require 'mpigenerator'
require 'boostgenerator'
require 'datatype'

class TypemapGenerator
  class << self
    # wraps both, the parsing and generation process. It's really just
    # for convenience.
    def generate_forest(xml_path,
                        template_path="./",
                        sloppy=false,
                        namespace=nil,
                        header_pattern=/^$/,
                        header_replacement="",
                        macro_guard_mpi=nil,
                        macro_guard_boost=nil)
      parser = MPIParser.new(xml_path, sloppy, namespace)

      mpi_generator = MPIGenerator.new(template_path, namespace, macro_guard_mpi)
      boost_generator = BoostGenerator.new(template_path, namespace, macro_guard_boost)

      mpi_classes = parser.find_classes_to_be_serialized("Typemaps").sort
      boost_classes = parser.find_classes_to_be_serialized("Serialization").sort

      mpi_options = parser.resolve_forest(mpi_classes)         + [header_pattern, header_replacement]
      boost_options = parser.shallow_resolution(boost_classes) + [header_pattern, header_replacement]

      mpi_ret = mpi_generator.generate_forest(*mpi_options)
      boost_ret = boost_generator.generate_forest(*boost_options)
      return boost_ret + mpi_ret
    end

    def find_classes_to_be_serialized(xml_path)
      parser = MPIParser.new(xml_path)
      classes = parser.find_classes_to_be_serialized
      return classes.map { |klass| parser.find_header(klass) }
    end
  end
end
