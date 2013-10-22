require 'pathname'

module BasicGenerator
  def init_generator(template_path="./", namespace=nil)
    @path = Pathname.new(template_path)
    @namespace = namespace

    if namespace
      @namespace_guard = namespace.upcase + "_"
      @namespace_begin = "namespace #{namespace} {\n"
      @namespace_end = "}\n"
    else
      @namespace_guard = ""
      @namespace_begin = ""
      @namespace_end = ""
    end
  end

  def guard(macro_guard, text)
    if macro_guard.nil?
      return text
    end
    return "#include<libgeodecomp/config.h>\n#ifdef #{macro_guard}\n#{text}\n#endif\n"
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
end
