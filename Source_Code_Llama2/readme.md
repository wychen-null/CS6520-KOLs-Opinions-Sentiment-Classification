# Llama2微调代码说明
Llama2微调部分代码基于 https://github.com/georgian-io/LLM-Finetuning-Toolkit 这个仓库的代码进行修改的，主要使用到了legacy分支中llama2微调部分代码。
请在运行代码前检查相应的文件夹目录是否被创建
### 数据获取
* 数据获取模块都在```prompt.py```中 
* 根据get_data_for_ft()方法中的注释去选择获取不同来源的数据 
* get_data_for_output()这个方法用于获取用于可靠性分析的数据
### 原始模型微调
1. ```llama2_classification.py```和```llama2_classification_inference.py```分别是对原生Llama2-7B模型进行微调以及对模型进行评估的代码
2. ```llama2_classification.py```将不同训练参数的模型输出到experiment目录下指定文件夹中<br>
   * 使用命令在后台对指定参数进行微调，设置lora_r、epochs和dropout等参数，并将日志保存到指定路径下
   ```shell
    nohup python llama2_classification.py --lora_r ${lora_r[$r]} --epochs ${epochs[$epoch]} --dropout ${dropout[$d]} --pretrained_ckpt <your model path> > ./log/output.log 2>&1 &
    ```
3. llama2_classification_inference.py对于2中输出路径下的所有模型进行评估
   * 使用命令对一个路径下所有模型进行评估
   ```shell
   nohup python llama2_classification_inference.py --experiment <your model path> > ./log/metric.log 2>&1 &
   ```
### 添加了一层MLP辅助降维
1. ```llama2_classification_mlp.py```和```llama2_classification_mlp_inference.py```是想Llama2-7B模型添加了一层mlp辅助降维的代码
2. 用命令对模型进行微调的代码与原始模型微调中的方法一致
### 运行脚本
1. ```run_lora.sh```是一个微调的脚本，通过设置里面的参数和命令，以遍历多个训练参数，用来找到最优调参。运行命令如下：
   ```shell
   nohup ./run_lora.sh > ./log/output.log 2>&1 &
   ```
2. 尾缀带有output的文件是用来输出可靠性分析的情感分类结果，命令使用参考inference为后缀的文件
3. ```Llama2-7B-conda-env.yaml```是进行Llama2微调的conda虚拟环境导出配置文件