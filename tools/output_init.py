from .output_tool import basic_output_function, null_output_function, output_function1, binary_output_function

output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "out1": output_function1,
    "binary": binary_output_function,
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
