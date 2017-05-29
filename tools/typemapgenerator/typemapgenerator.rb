load 'mpiparser.rb'
load 'mpigenerator.rb'
load 'boostgenerator.rb'
load 'hpxgenerator.rb'
load 'datatype.rb'

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
                        macro_guard_boost=nil,
                        macro_guard_hpx=nil,
                        include_prefix="")
      parser = MPIParser.new(xml_path, sloppy, namespace, include_prefix)

      mpi_generator = MPIGenerator.new(template_path, namespace, macro_guard_mpi)
      boost_generator = BoostGenerator.new(template_path, namespace, macro_guard_boost)
      hpx_generator = HPXGenerator.new(template_path, namespace, macro_guard_hpx)

      mpi_classes = parser.find_classes_to_be_serialized("Typemaps").sort
      boost_classes = parser.find_classes_to_be_serialized("BoostSerialization").sort
      hpx_classes = parser.find_classes_to_be_serialized("HPXSerialization").sort

      mpi_options = parser.resolve_forest(mpi_classes)
      mpi_options.header_pattern = header_pattern
      mpi_options.header_replacement = header_replacement

      boost_options = parser.shallow_resolution(boost_classes)
      boost_options.header_pattern = header_pattern
      boost_options.header_replacement = header_replacement

      hpx_options   = parser.shallow_resolution(hpx_classes)
      hpx_options.header_pattern = header_pattern
      hpx_options.header_replacement = header_replacement

      mpi_ret = mpi_generator.generate_forest(mpi_options)
      boost_ret = boost_generator.generate_forest(boost_options)
      hpx_ret = hpx_generator.generate_forest(hpx_options)

      return boost_ret + hpx_ret + mpi_ret
    end

    def find_classes_to_be_serialized(xml_path, friend_name)
      parser = MPIParser.new(xml_path)
      classes = parser.find_classes_to_be_serialized(friend_name)
      return classes.map { |klass| parser.find_header(klass) }
    end
  end
end
