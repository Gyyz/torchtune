# Finetune Step
This step is to finetune the pre-trained model on the downstream task of text-to-SQL.
currently, we only support finetuning on the Meta-Llama model.
1) git clone & install the torchtune package
2) download the Meta-Llama model
3) prepare the data
4) modify the config file
5) run the finetuning script

## 1. git clone the torchtune package
My folk:
```
git clone https://github.com/Gyyz/torchtune 
```
Official:
```
git clone https://github.com/pytorch/torchtune
```

## 2. install the torchtune package
```
cd torchtune
pip install -e .
```


## 3. prepare the data:
1) the data should be json exmaples with the following format:
```
{
    "instcution": 'insturction here',
    "input": 'your prompt here',
    "output": "your expected output here" 
}
```
2) you can modify the templete in `torchtune/data` to define the way to ensemble the data and define the templete you want to use in the config, you can also pass the templete as a parameter.

## 4. modify the config file (suppose the root dir text-to-sql-finetune)
1) copy the config file from the torchtune package to your project directory, you can find all the config files in `recipes/configs`.
`tune cp llama3_1/8B_full ./my_custom_config.yaml`
2) define the data path and the model type/path in the config file.
You can use huggingface datasets to load the data, or you can define your own data loading function in the config file.
For me, I create huggingface dataset to load my dataset:
The config file should look like this:
```
# Dataset
dataset:
  _component_: torchtune.datasets.alpaca_dataset
  source: 'gaoyzz/finetune_oneshot' # the huggingface dataset name
  split: 'train'
  data_files: 'train.json' # the name of the split
seed: null
shuffle: True
```

You can define to load local dataset, the config file should look like this:
```
# Dataset
dataset:
  _component_: torchtune.datasets.alpaca_dataset
  source: './spider/spider' # the dir to the data file
  # split: 'train'
  data_files: 'spider_oneshot_data.json' #the name of the data file
seed: null
shuffle: True
```
## 5. run the finetuning script
`--config` is the path to your personalized config file or the name of the config file in the `recipes/configs` directory.

For example, the config from torchtune(no need to include the file extension):
```
tune run full_finetune_distributed --config llama3/8B_full
```

the config from personalized config:
```
tune run full_finetune_distributed --config ./configs/llama3/spider/8B_full_oneshot.yaml
```

## 6. run generate script
The generate config should include the finetuned model path, the dataset file and the output file path.
you can modify the generate behavior in `recipes/generate.py`
```
tune run generate --config ./configs/llama3/spider/generation_zero.yaml
```