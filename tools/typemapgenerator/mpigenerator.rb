load 'basicgenerator.rb'

# Here we generate the C++ code that will in turn create the typemaps
# for MPI.
class MPIGenerator
  include BasicGenerator

  def initialize(template_path="./", namespace=nil, macro_guard_mpi=nil)
    @serialization_class_name = "Typemaps"
    init_generator(template_path, namespace)
    @macro_guard_mpi   = macro_guard_mpi
  end

  def simple_name(name)
    name.gsub(/[, :<>]+/, '_')
  end

  # returns the code of a method suitable to create a typemap for the
  # given klass.
  def generate_single_map(klass, members, parents)
    ret = File.read(@path + "template_generatesinglemap.cpp");

    ret.gsub!(/KLASS_NAME/, simple_name(klass))
    ret.gsub!(/KLASS/, klass)
    num_members = members.size
    num_members += parents.size unless parents.nil?
    ret.gsub!(/NUM_MEMBERS/, num_members.to_s)

    member_specs1 = members.map do |name, properties|
      "        MemberSpec(getAddress(&obj->#{name}), lookup<#{properties[:class]} >(), #{properties[:cardinality]})"
    end
    member_specs2 = []
    if parents
      member_specs2 = parents.map do |name, mpiname|
        "        MemberSpec(getAddress((#{name}*)obj), #{mpiname}, 1)"
      end
    end
    member_specs = member_specs1 + member_specs2
    ret.sub!(/ *MEMBERSPECS/, member_specs.sort.join(",\n"))
  end

  # The Typemap Class needs a header file, declaring all the static
  # variables, macros and so on. This method will generate it's code.
  def generate_header(options)
    ret = File.read(@path + "template_typemaps.h");
    ret.gsub!(/HEADERS/, map_headers(options.headers, options.header_pattern, options.header_replacement))
    ret.gsub!(/NAMESPACE_GUARD/, @namespace_guard)
    ret.gsub!(/NAMESPACE_BEGIN\n/, @namespace_begin)
    ret.gsub!(/NAMESPACE_END\n/, @namespace_end)

    class_vars = options.topological_class_sortation.map do |klass|
      klass_name = options.datatype_map[klass]
      "extern MPI_Datatype #{klass_name};"
    end
    ret.sub!(/.*CLASS_VARS/, class_vars.join("\n"))

    mapgens = options.topological_class_sortation.map do |klass|
      "    static MPI_Datatype generateMap#{simple_name(klass)}();"
    end
    ret.sub!(/.*MAPGEN_DECLARATIONS/, mapgens.join("\n"))

    lookup_types = Datatype.new.find_all { |k, v| v != :ignore }
    lookup_types = lookup_types.map {|k, v| k }
    lookup_types = (lookup_types.sort + options.topological_class_sortation).uniq

    lookups = lookup_types.map do |klass|
      <<EOF
    static inline MPI_Datatype lookup(#{klass}*)
    {
        return #{options.datatype_map[klass]};
    }
EOF
    end
    ret.sub!(/.*LOOKUP_DEFINITIONS/, lookups.join("\n"))

    return guard(@macro_guard_mpi, ret)
  end

  # The Typemap Class needs a source file, containing all the method
  # bodys and variable definitions. This methods creates the code.
  def generate_source(options)
    ret = File.read(@path + "template_typemaps.cpp");

    class_vars = options.topological_class_sortation.map do |klass|
      klass_name = options.datatype_map[klass]
      "MPI_Datatype #{klass_name};"
    end
    ret.sub!(/ *CLASS_VARS/, class_vars.join("\n"))
    ret.sub!(/NAMESPACE_BEGIN\n/, @namespace_begin)
    ret.sub!(/NAMESPACE_END\n/, @namespace_end)

    methods = options.topological_class_sortation.map do |klass|
      generate_single_map(klass,
                          options.resolved_classes[klass],
                          options.resolved_parents[klass])
    end
    ret.sub!(/METHOD_DEFINITIONS/, methods.join("\n"))

    assignments = options.topological_class_sortation.map do |klass|
      "    #{options.datatype_map[klass]} = generateMap#{simple_name(klass)}();"
    end
    ret.sub!(/.+ASSIGNMENTS/, assignments.join("\n"))

    if @macro_guard_mpi
      return guard(@macro_guard_mpi, ret)
    end

    return ret
  end

  # wraps the code generation for multiple typemaps.
  def generate_forest(options)
    return [generate_header(options),
            generate_source(options)]
  end
end
