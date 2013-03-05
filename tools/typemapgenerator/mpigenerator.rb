require 'pathname'

# Here we generate the C++ code that will in turn create the typemaps
# for MPI.
class MPIGenerator  
  def initialize(template_path="./", namespace=nil, macro_guard=nil)
    @path = Pathname.new(template_path)
    if namespace
      @namespace_guard = namespace.downcase + "_"
      @namespace_begin = "namespace #{namespace} {\n"
      @namespace_end = "};\n"
    else
      @namespace_guard = ""
      @namespace_begin = ""
      @namespace_end = ""
    end

    @macro_guard = macro_guard
  end

  def simple_name(name)
    name.gsub(/[ :<>]+/, '_')
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
      "        MemberSpec(MPI::Get_address(&obj->#{name}), #{properties[:type]}, #{properties[:cardinality]})"
    end
    member_specs2 = []
    if parents
      member_specs2 = parents.map do |name, mpiname|
        "        MemberSpec(MPI::Get_address((#{name}*)obj), #{mpiname}, 1)"
      end
    end
    member_specs = member_specs1 + member_specs2
    ret.sub!(/ *MEMBERSPECS/, member_specs.sort.join(",\n"))
  end

  # The Typemap Class needs a header file, declaring all the static
  # variables, macros and so on. This method will generate it's code.
  def generate_header(classes, datatype_map, headers, header_pattern=nil, header_replacement=nil)
    ret = File.read(@path + "template_typemaps.h");
    ret.gsub!(/HEADERS/, map_headers(headers, header_pattern, header_replacement))
    ret.gsub!(/NAMESPACE_GUARD/, @namespace_guard)
    ret.gsub!(/NAMESPACE_BEGIN\n/, @namespace_begin)
    ret.gsub!(/NAMESPACE_END\n/, @namespace_end)

    class_vars = classes.map do |klass|
      klass_name = datatype_map[klass].sub(/MPI::/, "")
      "    extern Datatype #{klass_name};"
    end
    ret.sub!(/.*CLASS_VARS/, class_vars.join("\n"))
    
    mapgens = classes.map do |klass|
      "    static MPI::Datatype generateMap#{simple_name(klass)}();"
    end
    ret.sub!(/.*MAPGEN_DECLARATIONS/, mapgens.join("\n"))
    
    lookup_types = (Datatype.new.keys.sort + classes).uniq
    lookups = lookup_types.map do |klass|
      "    static inline MPI::Datatype lookup(#{klass}*) { return #{datatype_map[klass]}; }"
    end
    ret.sub!(/.*LOOKUP_DEFINITIONS/, lookups.join("\n"))

    if @macro_guard
      return guard(ret)
    end

    return ret
  end

  def map_headers(headers, header_pattern, header_replacement)
    h = headers.map do |header|
      header_name = header
      if !header_replacement.nil?
        header_name = header.gsub(header_pattern, header_replacement)
      end
      "#include <#{header_name}>"
    end
    return h.join("\n")
  end

  # The Typemap Class needs a source file, containing all the method
  # bodys and variable definitions. This methods creates the code.
  def generate_source(topological_class_sortation, datatype_map, resolved_classes, resolved_parents)
    ret = File.read(@path + "template_typemaps.cpp");

    class_vars = topological_class_sortation.map do |klass|
      klass_name = datatype_map[klass].sub(/MPI::/, "")
      "    Datatype #{klass_name};"
    end
    ret.sub!(/ *CLASS_VARS/, class_vars.join("\n"))
    ret.sub!(/NAMESPACE_BEGIN\n/, @namespace_begin)
    ret.sub!(/NAMESPACE_END\n/, @namespace_end)

    methods = topological_class_sortation.map do |klass|
      generate_single_map(klass, resolved_classes[klass], resolved_parents[klass])
    end
    ret.sub!(/METHOD_DEFINITIONS/, methods.join("\n"))

    assignments = topological_class_sortation.map do |klass|
      "    #{datatype_map[klass]} = generateMap#{simple_name(klass)}();"
    end
    ret.sub!(/.+ASSIGNMENTS/, assignments.join("\n"))

    if @macro_guard
      return guard(ret)
    end

    return ret;
  end

  # wraps the code generation for multiple typemaps.
  def generate_forest(resolved_classes, resolved_parents, datatype_map, topological_class_sortation, headers, header_pattern=nil, header_replacement=nil)
    return [generate_header(topological_class_sortation, datatype_map, headers, header_pattern, header_replacement), 
            generate_source(topological_class_sortation, datatype_map, resolved_classes, resolved_parents)]
  end

  def guard(ret)
    return "#include<libgeodecomp/config.h>\n#ifdef #{@macro_guard}\n#{ret}\n#endif\n"
  end
end
