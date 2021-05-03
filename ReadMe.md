## Lawformer

#### Introduction
This repository provides the source code and checkpoints of the paper "Lawformer: A Pre-trained Language Model forChinese Legal Long Documents". 

#### Installation
```
pip install -r requirements.txt
```

#### Easy Start
We have uploaded our model to the huggingface model hub. Make sure you have installed transformers.
```python
>>> from transformers import AutoModel, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
>>> model = AutoModel.from_pretrained("xcjthu/Lawformer")
>>> inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
>>> outputs = model(**inputs)
```

#### Pre-training
We pre-train Lawformer continuously from `hfl/chinese-roberta-wwm-ext`. Therefore, we first convert the RoBERTa model to the Longformer by running the following command:
```
python3 convert_roberta_lfm.py
```
Then run the following command to pre-train the model:
```
python3 -m torch.distributed.launch --master_port 10086 --nproc_per_node 8 train.py -c config/Lawformer.config -g 0,1,2,3,4,5,6,7
```

#### Cite
If you use the pre-trained models, please cite this paper:
```
@article{xiao2021lawformer,
  title={Lawformer: A Pre-trained Language Model forChinese Legal Long Documents},
  author={Xiao, Chaojun and Hu, Xueyu and Liu, Zhiyuan and Tu, Cunchao and Sun, Maosong},
  year={2021}
}
```

