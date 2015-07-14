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

  def generate_serialize_function(klass, members, parents, template_parameters)
    params1 = render_template_params1(template_parameters)
    params2 = render_template_params2(template_parameters)

    ret = <<EOF
    template<typename ARCHIVE#{params1}>
    inline
    static void serialize(ARCHIVE& archive, #{klass}#{params2}& object, const unsigned /*version*/)
    {
EOF

    parents.sort.each do |parent_type|
      ret += <<EOF
        archive & #{base_object_name}<#{parent_type} >(object);
EOF
    end

    members.keys.sort.each do |member|
      ret += <<EOF
        archive & object.#{member};
EOF
    end

    ret += <<EOF
    }
EOF

    return ret
  end

  def render_template_params1(template_parameters)
    params = ""
    template_parameters.each do |parameter|
      params += ", #{parameter[:type]} #{parameter[:name]}"
    end

    return params
  end

  def render_template_params2(template_parameters)
    params = template_parameters.map { |parameter| parameter[:name] }
    params = params.join(", ")
    if (params.size > 0)
      params = "<#{params}>"
    end
  end
end
