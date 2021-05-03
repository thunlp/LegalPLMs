from .Lawformer import Lawformer

model_list = {
    "Lawformer": Lawformer,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
