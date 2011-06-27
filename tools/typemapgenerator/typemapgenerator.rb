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
                        namespace=nil)
      parser = MPIParser.new(xml_path, sloppy, namespace)
      generator = MPIGenerator.new(template_path, namespace)
      classes = parser.find_classes_to_be_serialized.sort
      return generator.generate_forest(*parser.resolve_forest(classes))
    end

    def find_classes_to_be_serialized(xml_path)
      parser = MPIParser.new(xml_path)
      classes = parser.find_classes_to_be_serialized
      return classes.map { |klass| parser.find_header(klass) }
    end
  end
end
