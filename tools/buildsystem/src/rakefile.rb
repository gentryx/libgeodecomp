require 'ostruct'
require 'pathname'
require 'set'
require 'yaml'

@configfile = "conf.yaml"
File.open(@configfile) do |f|
  @opts = YAML.load(f)
end

class Pathname
  # Returns true if path lies below directory
  def below?(directory)
    return to_s =~ /\A#{Pathname.new(directory)}/ ? true : false
  end
end

def test_header_pattern
  "#{@opts.srcdir}/**/test/*/*test.h"
end

def regen_typemaps
  puts "regenerating Typemaps"
  generate_documentation
  namespace = "--namespace #{@opts.package_namespace}"
  sh "ruby '#{@opts.srcdir}/../tools/typemapgenerator/generate.rb' #{namespace} --header-fix ^.+/src:#{@opts.package_namespace.downcase} ../doc/xml '#{@opts.typemapsdir}'"
end

# handle filenames with spaces safely (for cxxtest it isn't enough to
# enclose a path in quotes...) spaces are common when building on
# Windows, so this is no arcane feature.
def escape_pathname(path)
  path.to_s.gsub(/ /, "\\\\ ")
end

# check options:

required_opts = [:srcdir, :builddir, :cxxtestdir, :doxygen, :make, :cmake, :allowed_tests, :package_namespace]
required_opts += [:mpiexec] if @opts.mpi
required_opts += [:typemapsdir] if @opts.typemaps
required_opts.each do |opt|
  fail "Required option #{opt} not set." if @opts.send(opt).nil?
end

paths = [:srcdir, :builddir, :cxxtestdir]
paths += [:typemapsdir] if @opts.typemaps
paths.each do |path|
  fail "Option #{path} needs to be an instance of Pathname." if !(@opts.send(path).class == Pathname)
end

system "#{@opts.cmake} --version >/dev/null"
raise "Could not run CMAKE #{$?}" unless $?.success?

system "#{@opts.doxygen} --version >/dev/null"
raise "Could not run DoxyGen" unless $?.success?

# target chaining:

task :default => :compile

task :test => :unit

task :unit => :compile

task :compile => :cmake_prep

task :cmake_prep => :code_generation

task :code_generation => @opts.builddir.to_s

task :install => :compile do
  cd @opts.builddir
  sh "#{@opts.make} #{@opts.makeopts} install" 
end

task :code_generation => :typemaps 

directory @opts.builddir.to_s

# unit test generation:

clobber_list = []

unittests = FileList[test_header_pattern]
unittests.each do |header|
  source = header.sub(/.h$/, ".cpp")
  clobber_list.push source
  task :code_generation => source
  file source => header do
    header_name = escape_pathname(Pathname.new(header).basename)
    source_name = escape_pathname(Pathname.new(source).basename)
    cd Pathname.new(header).parent
    sh "'#{@opts.cxxtestdir}/cxxtestgen.pl' -o '#{source_name}' --part '#{header_name}'"
  end
end

testdirs = unittests.map do |header| 
  Pathname.new(header).parent
end
testdirs.uniq!

testdirs.each do |dir|
  unitrunner_source = dir + "main.cpp"
  clobber_list.push unitrunner_source
  task :code_generation => unitrunner_source
  file unitrunner_source do
    sh "'#{@opts.cxxtestdir}/cxxtestgen.pl' -o '#{escape_pathname(unitrunner_source)}' --root --error-printer"
  end

  if (unitrunner_source.below?(Rake.original_dir))
    task :unit do
      exeprefix = ""
      exeprefix = "#{@opts.mpiexec} -np #$1" if /parallel_mpi_(\d+)/ =~ unitrunner_source
      offset_path = unitrunner_source.parent.relative_path_from(@opts.srcdir)
      unitrunner_exe = @opts.builddir + offset_path + "test"
      if File.exists?(unitrunner_exe)
        sh "#{exeprefix} '#{unitrunner_exe}'"
      end
    end
  end
end

# cmake preparation (find sources, create source list files)
task :cmake_prep do
  dirmap = Hash.new([])
  all_sources.each do |file|
    pathname = Pathname.new(file)
    dir = pathname.parent
    dirmap[dir] = dirmap[dir] + [pathname]
  end

  dirmap.each do |dir, files|
    header_paths = files.select { |f| f.fnmatch("*.h") }
    source_paths = files.select { |f| f.fnmatch("*.cpp") }
    source_paths += files.select { |f| f.fnmatch("*.cu") }
    header_paths.sort!
    source_paths.sort!
    headers = header_paths.map { |f| "  ${RELATIVE_PATH}#{f.basename}" }
    sources = source_paths.map { |f| "  ${RELATIVE_PATH}#{f.basename}" }
    headers = headers.join("\n")
    sources = sources.join("\n")

    rendered = 
      "set(SOURCES ${SOURCES} \n" + sources + "\n)\n" +
      "set(HEADERS ${HEADERS} \n" + headers + "\n)\n"

    pri_name = dir + "auto.cmake"
    cached = IO.read(pri_name) rescue ""
    if cached != rendered
      puts "updating source list #{pri_name}"
      File.open(pri_name, "w") do |priFile|
        priFile.puts rendered
      end
    end
  end
end

# typemaps:
if !@opts.typemaps
  task :typemaps do
    puts "skipping Typemap generation"
  end
else
  @headercache = @opts.builddir + "headercache"
  @typemaps_header = @opts.typemapsdir + "typemaps.h"
  @typemaps_source = @opts.typemapsdir + "typemaps.cpp"
  typemaps_deps = []
  if File.exists?(@headercache)
    typemaps_deps = File.read(@headercache).split("\n")
  end

  task :typemaps => [@headercache, @typemaps_header, @typemaps_source]

  file @typemaps_header => @typemaps_source
  
  file @typemaps_source => typemaps_deps do  
    regen_typemaps
  end
  
  headers = FileList["#{@opts.srcdir}/**/*.h"].exclude(test_header_pattern).exclude(@typemaps_header)
  file @headercache => headers do |p|
    # at least create a empty file to indicate that there are no
    # classes to be serialized yet...
    unless File.exists?(p.name)  
      touch p.name 
      # ...but fake the access time so all headers are considered newer.
      File.utime(Time.at(0), Time.at(0), p.name)
    end
  
    cache_age = File.mtime(p.name)
    newer_headers = headers.find_all { |h| File.mtime(h) > cache_age }

    buffer = []
    if newer_headers.size == 1
      h = newer_headers[0]
      `grep 'friend class Typemaps;' #{h} 2>&1`
      buffer = [h] if $?.success?
    elsif newer_headers.size > 1
      buffer = `grep "^.*friend class Typemaps;" #{newer_headers.join(" ")} 2>/dev/null` 

      buffer = buffer.split("\n").map do |h| 
        h =~ /^(.+)\: /
        $1
      end
    end

    new_typemaps_deps = (buffer + typemaps_deps).uniq.sort
    if typemaps_deps != new_typemaps_deps    
      File.open(@headercache, "w") { |f| f.puts new_typemaps_deps.join("\n")}
      # instantly rebuild typemaps
      regen_typemaps
    end
  end
end

def all_sources
  FileList[@opts.srcdir + "**/*.{h,cpp,cu}"]
end

def generate_documentation
  cd @opts.srcdir
  sh "doxygen doxygen.conf"
end

task :doc do
  generate_documentation
end

task :compile => "#{@opts.builddir}/Makefile" do
  cd @opts.builddir
  sh "#{@opts.make} #{@opts.makeopts}" 
end

task :installer => :compile do
  cd @opts.builddir
  sh "#{@opts.make} #{@opts.makeopts} package"  
end

file "#{@opts.builddir}/Makefile" => [:cmake_prep, @opts.builddir] do
  # use relative paths so that we can use a Windows cmake version from
  # cygwin without having to convert full paths
  src_relative = @opts.srcdir.relative_path_from(@opts.builddir)
  cd @opts.builddir
  sh "#{@opts.cmake} #{src_relative}"
end

task :clean do
  auto_cmakes = Set.new
  all_sources.each do |file|
    pathname = Pathname.new(file)
    auto_cmakes.add(pathname.parent + "auto.cmake")
  end
  all_files = auto_cmakes.to_a + clobber_list +
    [@opts.srcdir + "../doc", 
     @opts.builddir, 
     @typemaps_source, 
     @typemaps_header, 
     @headercache]
  
  all_files.compact!
  all_files.each { |f| rm_rf f }
end

task :distclean => :clean do
  all_files = [@configfile, "conf.cmake"]
  all_files += @opts.distclean_files if @opts.distclean_files
  
  all_files.compact!
  all_files.each { |f| rm_rf f }
end
