# -*- coding: utf-8 -*-
# Copyright (C) 2006,2007 Andreas Schaefer <gentryx@gmx.de>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301 USA.

require 'rexml/document'
require 'logger'
require 'set'
require 'datatype'
require 'pp'

# This class is responsible for extracting all the information we need
# from Doxygen's XML output.
class MPIParser
  attr_accessor :datatype_map
  attr_accessor :type_hierarchy_closure

  # All doxygen xml files are expected in path, setting sloppy to true
  # will allow you to create partial typemaps (which simply ignore
  # members for which no MPI datatype could be generated). Be aware
  # that this won't work if the to be excluded members are of a
  # template type for whitch typemaps will be generated using other
  # parameters. Yeah, it's complicated.
  def initialize(path="../../../trunk/doc/xml", sloppy=false, namespace="")
    @path, @sloppy, @namespace = path, sloppy, namespace
    @log = Logger.new(STDOUT)
    @log.level = Logger::DEBUG

    class_files = Dir.glob("#{@path}/*.xml")
    @xml_docs = { }
    class_files.each do |filename|
      doc = REXML::Document.new File.new(filename)
      @xml_docs[filename] = doc
    end

    @filename_cache = { }
    @xml_docs.each do |filename, doc|    
      next if !is_class_declaration(filename)

      xpath = "doxygen/compounddef/compoundname"
      klass = parse_class_name(doc.elements[xpath].text)
      @filename_cache[klass] = filename
    end
    puts "filename_cache: "
    pp @filename_cache

    @datatype_map = Datatype.new
    @datatype_map.merge!(map_enums)
    classes_to_be_serialized = find_classes_to_be_serialized
    @type_hierarchy_closure = @datatype_map.keys.to_set +
      classes_to_be_serialized 
    @all_classes = classes_to_be_serialized 
  end

  # tries to resolve all datatypes given in classes to MPI type. For
  # those classes, whose MPI type could not be found in @datatype_map,
  # it'll try to create a new MPI type map specification.
  def resolve_forest(classes)
    puts "resolve_forest()"
    pp classes

    classes = classes.sort
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    @type_hierarchy_closure = @type_hierarchy_closure.union(classes)

    while classes.any?
      puts "  classes:"
      print "  "
      pp classes

      num_unresolved = classes.size

      classes.each do |klass|
        resolve_class(klass, classes,
                      resolved_classes, resolved_parents,
                      topological_class_sortation)
      end

      # fail if no class could be resolved in the last iteration
      if num_unresolved == classes.size
        raise "incomplete type hierarchy: could not resolve any in " +
          classes.inspect
      end
    end

    headers = topological_class_sortation.map { |klass| find_header(klass) }

    return [resolved_classes, resolved_parents, 
            @datatype_map, topological_class_sortation, headers]
  end

  def template_parameters(klass)
    xpath = "doxygen/compounddef/templateparamlist/param/declname"
    doc = @xml_docs[@filename_cache[klass]]
    
    template_params = []
    doc.elements.each(xpath) do |spec|
      template_params.push spec.text
    end

    return template_params
  end

  def used_template_parameters(klass)
    @log.debug "used_template_parameters(#{klass})"
    params = []
    klass =~ /^(#@namespace::|)(.+)/
    class_name = $2

    @all_classes.each do |c|
      c_template_params = template_parameters(c)
      members = get_members(c)

      members.each do |name, spec|
        puts "  - name: #{name}"
        puts "    spec: "
        pp spec

        if spec[:type] =~ /^(#@namespace::|)#{class_name}<(.+)>/
          # this will fail for constructs like Foo<Bar<int,int>,int>
          values = $2.split(",")
          values.map! { |v| v.strip }

          res = values.any? do |v|
            c_template_params.include?(v)
          end

          params.push values if !res
        end
      end
    end

    @log.debug "used_template_parameters returns"
    pp params.sort.uniq

    return params.sort.uniq
  end

  def map_template_parameters(members, template_params, values)
    param_map = { }
    values.size.times do |i|
      param_map[template_params[i]] = values[i]
    end

    new_members = { }
    members.each do |name, spec|
      new_spec = spec.clone

      param_map.each do |param, val|
        new_spec[:type] = new_spec[:type].gsub(/#{param}/, val)
      end

      new_members[name] = new_spec
    end

    return new_members
  end

  # wraps the resolution process (mapping of members) for a single class.
  def resolve_class(klass, classes,
                    resolved_classes, resolved_parents,
                    topological_class_sortation)
    begin
      members = get_members(klass)
      parents = get_parents(klass)

      template_params = template_parameters(klass)

      puts "----------------------------------"
      puts "resolve_class(#{klass})"
      puts "members"
      pp members
      puts "parents"
      pp parents
      puts "resolved_classes #{resolved_classes.size}"
      pp resolved_classes
      puts "template_params"
      pp template_params
      puts "----------------------------------"
      puts

      if template_params.empty?
        resolve_class_simple(klass, members, parents,
                             classes,
                             resolved_classes, resolved_parents,
                             topological_class_sortation)
      else
        used_params = used_template_parameters(klass)

        puts "used_params"
        pp used_params
        puts

        used_params.each do |values|
          new_members = 
            map_template_parameters(members, template_params, values)
          new_class = "#{klass}<#{values.join(",")} >"
          resolve_class_simple(new_class, new_members, parents,
                               classes,
                               resolved_classes, resolved_parents,
                               topological_class_sortation)
        end
      end

      classes.delete(klass)
    rescue Exception => e
      puts "failed with"
      pp e
      puts e.backtrace
    end
    # puts
  end

  def prune_unresolvable_members(members)
    ret = {}

    members.each do |klass, spec|
      next if @sloppy && exclude?(spec[:type])
      ret[klass] = spec
    end

    return ret
  end

  # fixme: refactor this shitty interface
  def resolve_class_simple(klass, members, parents, classes,
                           resolved_classes, resolved_parents,
                           topological_class_sortation)
    @log.debug("resolve_class_simple(#{klass})")

    actual_members = prune_unresolvable_members(members)
    member_map  = map_types_to_MPI_Datatypes(actual_members)
    parents_map = map_parent_types_to_MPI_Datatypes(parents)

    classes.delete klass
    topological_class_sortation.push klass
    @datatype_map[klass] = Datatype.cpp_to_mpi(klass, partial?(members))
    resolved_classes[klass] = member_map
    resolved_parents[klass] = parents_map
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
    members = { }

    sweep_all_members(klass) do |member| 
      klass, spec = parse_member(member)
      members[klass] = spec
    end

    return members
  end

  # returns an array containing all parent classes.
  def get_parents(klass)
    filename = class_to_filename(klass)
    doc = @xml_docs[filename]
    xpath = "doxygen/compounddef/basecompoundref"
    parents = []
    doc.elements.each(xpath) do |member| 
      stripped_member = member.text.gsub(/<\s*/, "<")
      puts "get_parents(#{klass} yields »#{stripped_member}«"
      parents.push stripped_member
    end   
    return parents
  end

  def lookup_type(type)
    return @datatype_map[type] || @datatype_map["#{@namespace}::#{type}"]
  end

  # tries to map all members to mpi datatypes (using datatype_map as a
  # dictionary). Returns nil if a type could not be found.
  def map_types_to_MPI_Datatypes(members)
    resolved = { }
    
    members.each do |name, map_orig| 
      lookup = lookup_type(map_orig[:type])
      unless lookup
        name1 = map_orig[:type]
        name2 = "#{@namespace}::#{map_orig[:type]}"
        raise "could not resolve member #{name1} or #{name2}"
      end

      map = map_orig.clone
      map[:type] = lookup
      resolved[name] = map
    end
    return resolved
  end

  # tries to map all parent types to mpi datatypes. Returns nil if a
  # type could not be found. 
  def map_parent_types_to_MPI_Datatypes(parents)
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
    @log.debug("parse_member(#{name})")
    puts "---------member"
    p member
    pp member
    puts member

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
  # Cannot handle arrays with non-symbolic width (e.g. "int bar[69]"),
  # and I refuse an implementation until we have a use case for that.
  def resolve_cardinality(member)
    @log.debug "resolve_cardinality(#{member})"

    argsString = member.elements["argsstring"]

    # easy case: non-array member
    return 1 unless argsString.has_text?
    # more difficult: array members...
    raise "illegal cardinality" unless argsString.text =~ /\[(.+)\]/ 
    # numeric constant as array size:
    return $1.to_i if argsString.text =~ /\[(\d+)\]/
    # gotta search a little longer for symbolic sizes:
    @log.debug "  non-trivial cardinality"
    member_id = member.attributes["id"]
    cardinality_id = resolve_cardinality_id(member_id)
    return resolve_cardinality_declaration(cardinality_id)
  end

  # Doxygen assigns items an ID. The cardinality of an array member is
  # an item. Unfortunately Doxygen's XML output in files named
  # "classXXX.xml" lacks such a reference ID. Nevertheless, we can dig
  # for it in the files "XXX_8h.xml".
  def resolve_cardinality_id(member_id)
    @log.debug "resolve_cardinality_id(#{member_id})"

    unless member_id =~ /class#{@namespace}(_1_1)*(.+)_.+/
      raise "illegal member ID"
    end

    klass = $2.downcase
    filename = "#{@path}/#{klass}_8h.xml"

    @log.debug "opening #{filename}, namespace: #{@namespace}"
    doc = @xml_docs[filename]

    codeline = nil
    doc.elements.each("doxygen/compounddef/programlisting/codeline") do |line|
      @log.debug "  codeline #{line}"
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
      # node = elem.find do |elem| 
        puts "    elemelem: #{elem}"
        puts "    elemelem.class: #{elem.class}"
        if elem.class == REXML::Element
          refid = elem.attributes["refid"]
          puts "    refid: #{refid}"
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
    @log.debug "resolve_cardinality_declaration(#{cardinality_id})"

    sweep_all_classes do |klass, member|
      if member.attributes["id"] == cardinality_id
        name = member.elements["name"].text
        return "#{klass}::#{name}"
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
    @log.debug "  filename = #{filename}"

    doc = @xml_docs[filename]
    xpath = "doxygen/compounddef/sectiondef/memberdef[@kind='#{kind}'][@static='no']"

    doc.elements.each(xpath) do |member|
      yield member
    end   
  end

  # locate the header filename
  def find_header(klass)
    begin
        find_header_simple(klass)
    rescue Exception => e
      if klass =~ /<.+>/
        find_header_simple(template_basename(klass))
      else
        raise e
      end
    end
  end

  def find_header_simple(klass)
    puts "find_header_simple(#{klass})"

    filename = class_to_filename(klass)
    doc = @xml_docs[filename]
    xpath = "doxygen/compounddef/location"

    doc.elements.each(xpath) do |member|
      header = member.attributes["file"]
      puts "  header: #{header}"
      return header if header
    end   

    raise "no header found for class #{klass}"
  end

  def find_classes_to_be_serialized
    ret = Set.new

    sweep_all_classes do |klass, member|
      if member.attributes["kind"] == "friend" && 
          member.elements["name"].text == "Typemaps"
        ret.add klass
      end      
    end

    return ret
  end

  def is_class_declaration(filename)
    (filename =~ /\/class/)
  end

  def class_to_filename(klass)
    @filename_cache[klass]
  end

  def parse_class_name(klass)
    ret = klass.gsub(/ /, "")
    return ret.gsub(/>/, " >")
  end
end
