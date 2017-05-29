# -*- coding: utf-8 -*-
require 'rexml/document'
require 'logger'
require 'ostruct'
require 'pp'
require 'set'
require 'stringio'
load 'datatype.rb'

# This class is responsible for extracting all the information we need
# from Doxygen's XML output.
class MPIParser
  attr_accessor :datatype_map
  attr_accessor :type_hierarchy_closure
  attr_accessor :log
  attr_accessor :filename_cache

  # All doxygen xml files are expected in path, setting sloppy to true
  # will allow you to create partial typemaps (which simply ignore
  # members for which no MPI datatype could be generated). Be aware
  # that this won't work if the to be excluded members are of a
  # template type for whitch typemaps will be generated using other
  # parameters. Yeah, it's complicated.
  def initialize(path="../../../trunk/doc/xml", sloppy=false, namespace="", include_prefix="")
    @path, @sloppy, @namespace = path, sloppy, namespace
    @log = Logger.new(STDOUT)
    @log.level = Logger::WARN
    # @log.level = Logger::INFO
    # @log.level = Logger::DEBUG
    @member_cache = {}

    class_files = grep_typemap_candidates(path)
    @xml_docs = { }
    @xml_cache = {}

    @include_prefix = include_prefix

    threads = []
    num_threads = 1
    slices = class_files.each_slice(num_threads)

    num_threads.times do |i|
      thread = Thread.new do
        slices.each do |slice|
          filename = slice[i]
          break if filename.nil?

          doc = REXML::Document.new File.new(filename)
          @xml_docs[filename] = doc
        end
      end

      threads.push thread
    end

    threads.each { |t| t.join }

    @filename_cache = { }
    @xml_docs.each do |filename, doc|
      next if !is_class_declaration(filename)

      xpath = "doxygen/compounddef/compoundname"
      klass = parse_class_name(doc.elements[xpath].text)
      @filename_cache[klass] = filename
    end

    @log.debug "filename_cache: "
    @log.debug pp(@filename_cache)

    @datatype_map = Datatype.new
    @datatype_map.merge!(map_enums)
    classes_to_be_serialized =
      find_classes_to_be_serialized("Typemaps") +
      find_classes_to_be_serialized("BoostSerialization") +
      find_classes_to_be_serialized("HPXSerialization")
    @type_hierarchy_closure = @datatype_map.keys.to_set + classes_to_be_serialized
    @all_mpi_classes = find_classes_to_be_serialized("Typemaps")
  end

  def grep_typemap_candidates(path)
    files = []

    ["Typemaps", "BoostSerialization", "HPXSerialization"].each do |klass|
      files += `grep "friend class #{klass}"             #{path}/class*.xml | cut -d : -f 1`.split("\n")
      files += `grep "<definition>#{klass}</definition>" #{path}/class*.xml | cut -d : -f 1`.split("\n")
    end

    return files.uniq
  end

  def get_xml(filename)
    if @xml_cache[filename].nil?
      doc = REXML::Document.new File.new(filename)
      @xml_docs[filename] = doc
    end

    return @xml_docs[filename]
  end

  def pp(object)
    buffer = StringIO.new
    PP.pp(object, buffer)
    return buffer.string
  end

  # tries to resolve all datatypes given in classes to MPI type. For
  # those classes, whose MPI type could not be found in @datatype_map,
  # it'll try to create a new MPI type map specification.
  def resolve_forest(classes)
    @log.info "resolve_forest()"
    @log.debug pp(classes)

    res = OpenStruct.new
    classes = classes.sort
    res.resolved_classes = { }
    res.resolved_parents = { }
    res.topological_class_sortation = []
    res.is_abstract = {}
    @type_hierarchy_closure = @type_hierarchy_closure.union(classes)

    while classes.any?
      @log.debug "  classes:"
      @log.debug pp(classes)
      @log.debug "  resolved_classes:"
      @log.debug pp(res.resolved_classes)
      num_unresolved = classes.size
      # this temporary clone is required to avoid interference with deleted elements
      temp_classes = classes.clone

      temp_classes.each do |klass|
        resolve_class(klass, classes,
                      res.resolved_classes, res.resolved_parents,
                      res.topological_class_sortation, res.is_abstract)
      end

      # fail if no class could be resolved in the last iteration
      if num_unresolved == classes.size
        raise "incomplete type hierarchy: could not resolve any in " +
          classes.inspect
      end
    end

    @log.info "  forest resolution successful, mapping headers"
    res.headers = res.topological_class_sortation.map do |klass|
      find_header(klass, @include_prefix)
    end
    res.datatype_map = @datatype_map
    return res
  end

  def shallow_resolution(classes)
    @log.info "shallow_resolution()"
    classes = classes.sort

    res = OpenStruct.new
    res.topological_class_sortation = classes.sort
    res.members = {}
    res.resolved_parents = {}
    res.template_params = {}
    res.is_abstract = {}
    res.wants_polymorphic_serialization = {}

    classes.each do |klass|
      res.members[klass] = get_members(klass)
      res.resolved_parents[klass] = get_parents(klass)
      res.template_params[klass] = template_parameters(klass)
      res.is_abstract[klass] = is_abstract?(klass)
      res.wants_polymorphic_serialization[klass] = wants_polymorphic_serialization?(klass)
    end

    @log.info "  shallow resolution successful, mapping headers"
    res.headers = classes.map { |klass| find_header(klass, @include_prefix) }
    return res
  end

  def template_parameters(klass)
    @log.debug "template_parameters(#{klass})"
    xpath = "doxygen/compounddef/templateparamlist/param"
    filename = @filename_cache[klass]
    doc = get_xml(filename)
    @log.debug "scouring file #{filename}"
    template_params = []

    doc.elements.each(xpath) do |spec|
      type = spec.get_elements("type")[0].text
      @log.debug "  type: #{type}"

      if type.nil?
        type = spec.get_elements("type")[0].get_elements("ref")[0].text
      end

      declname_elem = spec.get_elements("declname")[0]
      name = nil

      if declname_elem.nil?
        if type =~ /typename (\w+)/
          @log.debug "    path1"
          name = $1
          type = "typename"
        else
          if type.to_s =~ /^\s*typename\s*$/
            @log.debug "    path2"
            name = type.to_s
            type = "typename"
          else
            raise "failed to parse template parameter #{type} for class #{klass}"
          end
        end
      else
        name = declname_elem.text
      end

      s = {
        :name => name,
        :type => type
      }

      template_params.push s
    end

    @log.debug "template_parameters(#{klass}) done"
    @log.debug "template_parameters: #{template_params}"
    return template_params
  end

  def used_template_parameters(klass)
    @log.info "used_template_parameters(#{klass})"
    params = []
    klass =~ /^(#@namespace::|)(.+)/
    class_name = $2

    @all_mpi_classes.each do |c|
      @log.debug "used_template_parameters(#{klass}) -> #{c}"
      c_template_param_names = template_parameters(c).map do |param|
        param[:name]
      end
      members = get_members(c)

      members.each do |name, spec|
        @log.debug "  - name: #{name}"
        @log.debug "    type: #{spec[:type]}"
        @log.debug "    regex: ^(#@namespace::|)#{class_name}<(.+)>"
        @log.debug "    spec: "
        @log.debug pp(spec)

        if spec[:type] =~ /^(#@namespace::|)#{class_name}<(.+)>/
          @log.debug "  match!"
          # fixme: this will fail for constructs like Foo<Bar<int,int>,int>
          values = $2.split(",")
          values.map! { |v| v.strip }

          res = values.any? do |v|
            c_template_param_names.include?(v)
          end

          params.push values if !res
        end
      end
    end

    ret = params.sort.uniq
    @log.debug "used_template_parameters(#{klass}) returns"
    @log.debug pp(ret)
    return ret
  end

  def map_template_parameters(members, template_params, values)
    @log.info "map_template_parameters(
  members: #{pp members},
  template_params: #{pp template_params},
  values: #{pp values}
)"

    param_map = { }
    values.size.times do |i|
      param_map[template_params[i]] = values[i]
    end

    new_members = { }
    members.each do |name, spec|
      new_spec = spec.clone

      param_map.each do |param, val|
        pattern = "(^|\\W)(#{param})($|\\W)"
        replacement = "\\1#{val}\\3"
        new_spec[:type       ] = new_spec[:type       ].gsub(/#{pattern}/, replacement)
        new_spec[:cardinality] = new_spec[:cardinality].gsub(/#{pattern}/, replacement) if new_spec[:cardinality].class != Fixnum
      end

      new_members[name] = new_spec
    end

    return new_members
  end

  # wraps the resolution process (mapping of members) for a single class.
  def resolve_class(klass, classes,
                    resolved_classes, resolved_parents,
                    topological_class_sortation, is_abstract)
    begin
      members = get_members(klass)
      parents = get_parents(klass)

      template_params = template_parameters(klass).map do |param|
        param[:name]
      end

      @log.debug "----------------------------------"
      @log.info  "resolve_class(#{klass})"
      @log.debug "members"
      @log.debug pp(members)
      @log.debug "parents"
      @log.debug pp(parents)
      @log.debug "resolved_classes #{resolved_classes.size}"
      @log.debug pp(resolved_classes)
      @log.debug "template_params"
      @log.debug pp(template_params)
      @log.debug "----------------------------------"
      @log.debug ""

      if template_params.empty?
        @log.debug "  simple path"

        resolve_class_simple(klass, members, parents,
                             classes,
                             resolved_classes, resolved_parents,
                             topological_class_sortation, is_abstract)
      else
        used_params = used_template_parameters(klass)

        @log.debug "  used_params"
        @log.debug pp(used_params)

        used_params.each do |values|
          new_members =
            map_template_parameters(members, template_params, values)
          new_class = "#{klass}<#{values.join(",")} >"
          resolve_class_simple(new_class, new_members, parents,
                               classes,
                               resolved_classes, resolved_parents,
                               topological_class_sortation, is_abstract)
        end
      end

      classes.delete(klass)
    rescue Exception => e
      @log.info "failed with"
      @log.info pp(e)
      @log.info e.backtrace
    end
  end

  def prune_unresolvable_members(members)
    @log.debug("prune_unresolvable_members(#{members})")
    ret = {}

    members.each do |klass, spec|
      next if @sloppy && exclude?(spec[:type])
      ret[klass] = spec
    end

    return ret
  end

  def resolve_class_simple(klass, members, parents, classes,
                           resolved_classes, resolved_parents,
                           topological_class_sortation, is_abstract)
    @log.debug("resolve_class_simple(#{klass})")

    actual_members = prune_unresolvable_members(members)
    member_map  = map_types_to_MPI_Datatypes(actual_members)
    parents_map = map_parent_types_to_MPI_Datatypes(parents)

    classes.delete klass
    topological_class_sortation.push klass
    @datatype_map[klass] = Datatype.cpp_to_mpi(klass, partial?(members))
    resolved_classes[klass] = member_map
    resolved_parents[klass] = parents_map
    is_abstract[klass] = is_abstract?(klass)
  end

  def is_abstract?(klass)
    @log.debug "is_abstract?(#{klass})"

    begin
      sweep_all_functions(klass) do |member|
        next if member.attributes["virt"].nil?

        if member.attributes["virt"] == "pure-virtual" && member.attributes["ambiguityscope"].nil?
          return true
        end

      end
    rescue Exception => e
      @log.debug "  considering #{klass} as concrete, caught #{e}"
      return false
    end
    @log.debug "  #{klass} is concrete"
    return false
  end

  # checks if some class members will be excluded from serialization.
  def partial?(members)
    members.each do |klass, spec|
      return true if exclude?(spec[:type])
    end

    return false
  end

  # returns a map consisting of all member variables listed in the
  # class' doxygen .xml file.
  def get_members(klass)
    @log.debug "get_members(#{klass})"

    entry = @member_cache[klass]
    return entry unless entry.nil?

    members = { }

    sweep_all_members(klass) do |member|
      klass, spec = parse_member(member)
      members[klass] = spec
    end

    @member_cache[klass] = members
    return members
  end

  # returns an array containing all parent classes.
  def get_parents(klass)
    @log.info "get_parents(#{klass})"

    filename = class_to_filename(klass)
    doc = get_xml(filename)
    xpath = "doxygen/compounddef/basecompoundref"
    parents = []
    doc.elements.each(xpath) do |member|
      stripped_member = member.text.gsub(/<\s*/, "<")
      @log.debug "  »#{stripped_member}«"
      parents.push stripped_member
    end
    return parents
  end

  def lookup_type(type)
    return @datatype_map["#{type}"   ] || @datatype_map["#{@namespace}::#{type}"   ] ||
           @datatype_map["#{type}< >"] || @datatype_map["#{@namespace}::#{type}< >"]
  end

  # tries to map all members to mpi datatypes (using datatype_map as a
  # dictionary). Returns nil if a type could not be found.
  def map_types_to_MPI_Datatypes(members)
    @log.debug("map_types_to_MPI_Datatypes(#{members})")
    resolved = { }

    members.each do |name, map_orig|
      lookup = lookup_type(map_orig[:type])
      unless lookup
        name1 = map_orig[:type]
        name2 = "#{@namespace}::#{map_orig[:type]}"
        raise "could not resolve member #{name1} or #{name2}"
      end

      map = map_orig.clone
      map[:class] = map[:type]
      map[:type] = lookup
      resolved[name] = map
    end
    return resolved
  end

  # tries to map all parent types to mpi datatypes. Returns nil if a
  # type could not be found.
  def map_parent_types_to_MPI_Datatypes(parents)
    @log.debug("map_parent_types_to_MPI_Datatypes(#{parents})")

    resolved = { }
    parents.each do |name|
      lookup = lookup_type(name)
      unless lookup
        raise "could not resolve parent #{name} or #{@namespace + "::" + name}"
      end
      resolved[name] = lookup
    end
    return resolved
  end

  def template_basename(klass)
    klass =~ /([^<]+)(<.+>)*/
    $1
  end

  # Determine if a member variable may be excluded since we're
  # performing sloppy parsing and we're not able to create a datatype map ourselves.
  def exclude?(klass)
    # for exclusion, look for template name (without parameter list), too
    stripped = template_basename(klass)
    return @sloppy &&
      !@type_hierarchy_closure.include?(klass) &&
      !@type_hierarchy_closure.include?(stripped)
  end

  # is liable for extracting member information from the belonging XML node.
  def parse_member(member)
    name = member.elements["name"].text
    @log.info  "parse_member(#{name})"
    @log.debug "---------member"
    @log.debug member.to_s

    spec = {
      :type => extract_type(member),
      :cardinality => resolve_cardinality(member)
    }

    return [name, spec]
  end

  def extract_type(member)
    definition = member.elements.each("definition") do |definition|
      definition = definition.text
      definition.gsub!(/&lt;/, "<")
      definition.gsub!(/&gt;/, ">")

      index = 0
      depth = 0

      definition.size.times do |i|
        index = i
        char = definition[i..i]
        break if (char == " ") && (depth == 0)
        depth += 1 if char == "<"
        depth -= 1 if char == ">"
      end

      definition = definition[0...index]
      return parse_class_name(definition)
    end

    raise "could not extract type"
  end

  # gathers the cardinality for a member. It distinguishes simple
  # members (e.g. "int foo") from arrays (e.g. "int bar[Doedel]").
  def resolve_cardinality(member)
    @log.debug "resolve_cardinality(#{member.to_s[0...10]})"

    argsString = member.elements["argsstring"]

    # easy case: non-array member
    if !argsString.has_text?
      @log.debug("resolve_cardinality -> non-array member")
      return 1
    end

    # more difficult: array members...
    if !(argsString.text =~ /\[(.+)\]/)
      @log.debug("resolve_cardinality -> illegal cardinality found in member definition")
      raise "illegal cardinality"
    end

    # numeric constant as array size:
    if argsString.text =~ /\[(\d+)\]/
      @log.debug("resolve_cardinality -> numeric constant")
      return $1.to_i
    end

    # gotta search a little longer for symbolic sizes:
    @log.debug "resolve_cardinality -> non-trivial cardinality"

    # the following code seems over-engineered, so it got replaced by the simple regex below:
    #   member_id = member.attributes["id"]
    #   cardinality_id = resolve_cardinality_id(member_id)
    #   return resolve_cardinality_declaration(cardinality_id)

    argsString.text =~ /\[(.*)\]/
    return $1

  end

  def member_id_to_8h_file(member_id)
    member_id =~ /(class.+)_([^_]+)/
    filename = "#{@path}/#{$1}.xml"
    doc = get_xml(filename)
    doc.elements.each("doxygen/compounddef/includes") do |inc|
      eight_h_file = "#{@path}/#{inc.attributes["refid"]}.xml"
      return eight_h_file
    end

    raise "could not find _8h file"
  end

  # Doxygen assigns items an ID. The cardinality of an array member is
  # an item. Unfortunately Doxygen's XML output in files named
  # "classXXX.xml" lacks such a reference ID. Nevertheless, we can dig
  # for it in the files "XXX_8h.xml".
  def resolve_cardinality_id(member_id)
    @log.info "resolve_cardinality_id(#{member_id})"

    filename = member_id_to_8h_file(member_id)

    @log.debug "opening #{filename}, namespace: #{@namespace}"
    @log.debug pp member_id
    doc = get_xml(filename)

    codeline = nil
    doc.elements.each("doxygen/compounddef/programlisting/codeline") do |line|
      # @log.debug "  codeline #{line}"
      if line.attributes["refid"] == member_id
        codeline = line
        break
      end
    end

    @log.debug "selected codeline: #{codeline}"

    cardinality = nil
    codeline.elements.each("highlight") do |elem|
      @log.debug "  elem: #{elem}"

      elem.each do |elem|
        if elem.class == REXML::Element
          refid = elem.attributes["refid"]
          cardinality = refid unless refid.nil?
        end
      end
    end

    throw "failed to find cardinality for member_id #{member_id}" unless cardinality
    return cardinality
  end

  # uses the ID to identify the bit of code that makes up the
  # cardinality of an array.
  def resolve_cardinality_declaration(cardinality_id)
    @log.info "resolve_cardinality_declaration(#{cardinality_id})"

    sweep_all_classes do |klass, member|
      if member.attributes["id"] == cardinality_id
        name = member.elements["name"].text
        is_static = member.attributes["static"]

        @log.debug "  is_static: #{is_static}"
        @log.debug "  member:"
        @log.debug member.to_s

        # We need to distinguish between template parameters (e.g.
        # FloatCoord.vec) or numerical constants (e.g. Car.wheels).
        # The current solution is cheesy: if the cardinality is static
        # then we assume it to be a static const var, otherwise a
        # template parameter.
        if (is_static == "yes")
          ret = "#{klass}::#{name}"
        else
          member.elements["argsstring"].text =~ /\[(.+)\]/
          ret = $1
        end

        @log.debug "  returning #{ret}"
        return ret
      end
    end

    raise "cardinality declaration not found"
  end

  # locates all enumeration types contained in classes.
  def find_enums
    enums = []

    sweep_all_classes do |klass, member|
      if member.attributes["kind"] == "enum"
        enums.push member.elements["name"].text
      end
    end

    return enums
  end

  # creates a mapping of all enumeration types to the corresponding
  # int type for MPI. C++ handles enumeration variables like integers,
  # and so does MPI.
  def map_enums
    ret = { }
    int_MPI_type = lookup_type("int")
    find_enums.each { |enum| ret[enum] = int_MPI_type }
    return ret
  end

  # wraps the iteration though all XML class definitions and their
  # contained class members.
  def sweep_all_classes
    @xml_docs.each do |filename, doc|
      next if !is_class_declaration(filename)
      raw_name = doc.elements["doxygen/compounddef/compoundname"].text
      klass = parse_class_name(raw_name)
      doc.elements.each("doxygen/compounddef/sectiondef/memberdef") do |member|
        yield(klass, member)
      end
    end
  end

  # iterates though all members of klass class that are instance specific variables.
  def sweep_all_members(klass, kind="variable")
    @log.debug "sweep_all_members(#{klass})"
    filename = class_to_filename(klass)

    doc = get_xml(filename)
    xpath = "doxygen/compounddef/sectiondef/memberdef[@kind='#{kind}'][@static='no']"

    doc.elements.each(xpath) do |member|
      yield member
    end
  end

  def sweep_all_functions(klass)
    @log.debug "sweep_all_functions(#{klass})"
    filename = class_to_filename(klass)

    doc = get_xml(filename)
    xpath = "doxygen/compounddef/listofallmembers/member"

    doc.elements.each(xpath) do |member|
      yield member
    end
  end

  # locate the header filename
  def find_header(klass, prefix="")
    begin
      return prefix + find_header_simple(klass)
    rescue Exception => e
      @log.debug "failed to find header for #{klass}, caught #{e}, prefix: #{prefix.nil?}"
      if klass =~ /<.+>/
        prefix + find_header_simple(template_basename(klass))
      else
        raise e
      end
    end
  end

  def find_header_simple(klass)
    @log.info "find_header_simple(#{klass})"

    filename = class_to_filename(klass)
    doc = get_xml(filename)
    xpath = "doxygen/compounddef/location"

    doc.elements.each(xpath) do |member|
      header = member.attributes["file"]
      return header if header
    end

    raise "no header found for class #{klass}"
  end

  def find_classes_to_be_serialized(friend_name)
    ret = Set.new

    sweep_all_classes do |klass, member|
      if member.attributes["kind"] == "friend" &&
          member.elements["name"].text == friend_name
        ret.add klass
      end
    end

    return ret
  end

  def wants_polymorphic_serialization?(klass)
    filename = class_to_filename(klass)
    doc = get_xml(filename)

    doc.elements.each("doxygen/compounddef/sectiondef/memberdef") do |member|
      if (member.attributes["kind"] == "friend") &&
          (member.elements["name"].text == "PolymorphicSerialization")
        return true
      end
    end

    return false
  end

  def is_class_declaration(filename)
    (filename =~ /\/class/)
  end

  def class_to_filename(klass)
    klass =~ /([^\<]+)/
    stripped_class = $1
    res = @filename_cache[klass] || @filename_cache[stripped_class]
    if res.nil?
      throw "XML file name not found for class #{klass}"
    end
    return res
  end

  def parse_class_name(klass)
    ret = klass.gsub(/ /, "")
    return ret.gsub(/>/, " >")
  end
end
