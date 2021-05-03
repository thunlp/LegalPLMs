from transformers import LongformerForMaskedLM,RobertaForMaskedLM,AutoModelForMaskedLM,AutoTokenizer
import copy
import torch

max_pos = 4096
attention_window = 512

roberta = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", model_max_length=max_pos)

# extend position embedding
config = roberta.config
tokenizer.model_max_length = max_pos
tokenizer.init_kwargs['model_max_length'] = max_pos
current_max_pos, embed_size = roberta.bert.embeddings.position_embeddings.weight.shape
max_pos += 2
config.max_position_embeddings = max_pos
assert max_pos > current_max_pos

new_pos_embed = roberta.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
# copy position embeddings over and over to initialize the new position embeddings
k = 2
step = current_max_pos - 2
while k < max_pos - 1:
    if k + step >= max_pos:
        new_pos_embed[k:] = roberta.bert.embeddings.position_embeddings.weight[2:(max_pos + 2 - k)]
    else:
        new_pos_embed[k:(k + step)] = roberta.bert.embeddings.position_embeddings.weight[2:]
    k += step
roberta.bert.embeddings.position_embeddings.weight.data = new_pos_embed
roberta.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

# add global attention
config.attention_window = [attention_window] * config.num_hidden_layers
for i in range(len(roberta.bert.encoder.layer)):
    roberta.bert.encoder.layer[i].attention.self.query_global = copy.deepcopy(roberta.bert.encoder.layer[i].attention.self.query)
    roberta.bert.encoder.layer[i].attention.self.key_global = copy.deepcopy(roberta.bert.encoder.layer[i].attention.self.key)
    roberta.bert.encoder.layer[i].attention.self.value_global = copy.deepcopy(roberta.bert.encoder.layer[i].attention.self.value)

lfm = LongformerForMaskedLM(config)
lfm.longformer.load_state_dict(roberta.bert.state_dict())
lfm.lm_head.dense.load_state_dict(roberta.cls.predictions.transform.dense.state_dict())
lfm.lm_head.layer_norm.load_state_dict(roberta.cls.predictions.transform.LayerNorm.state_dict())
lfm.lm_head.decoder.load_state_dict(roberta.cls.predictions.decoder.state_dict())
lfm.lm_head.bias = copy.deepcopy(roberta.cls.predictions.bias)

lfm.save_pretrained('PLMConfig/roberta-converted-lfm')
tokenizer.save_pretrained('PLMConfig/roberta-converted-lfm')