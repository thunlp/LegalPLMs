from .accuracy_tool import single_label_top1_accuracy, single_label_top2_accuracy, multi_label_accuracy, \
    null_accuracy_function

accuracy_function_dic = {
    "SingleLabelTop1": single_label_top1_accuracy,
    "MultiLabel": multi_label_accuracy,
    "Null": null_accuracy_function
}


def init_accuracy_function(config, *args, **params):
    name = config.get("output", "accuracy_method")
    if name in accuracy_function_dic:
        return accuracy_function_dic[name]
    else:
        raise NotImplementedError
