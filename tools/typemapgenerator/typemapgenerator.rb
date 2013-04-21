require 'mpiparser'
require 'mpigenerator'
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
                        macro_guard=nil)
      parser = MPIParser.new(xml_path, sloppy, namespace)
      generator = MPIGenerator.new(template_path, namespace, macro_guard)
      classes = parser.find_classes_to_be_serialized.sort
      options = parser.resolve_forest(classes) + [header_pattern, header_replacement]
      return generator.generate_forest(*options) 
    end

    def find_classes_to_be_serialized(xml_path)
      parser = MPIParser.new(xml_path)
      classes = parser.find_classes_to_be_serialized
      return classes.map { |klass| parser.find_header(klass) }
    end
  end
end
