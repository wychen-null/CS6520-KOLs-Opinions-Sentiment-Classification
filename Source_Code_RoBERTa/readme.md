# RoBERTa-Large预训练+微调代码说明
这部分代码基于 https://github.com/Tinaaaqwq/ruoxinli3-22CS145 此仓库的微调代码进行修改<br>
* 在运行代码前需确认所需文件路径是否被创建
* 代码中的文件路径请根据实际的工程路径修改
### 代码说明
1. ```Further_pretraining.py```是用来进行预训练的代码，所使用的数据是<br>
```data/further_training_data/train.txt```
2. ```SMART_Roberta_Large_tweets_Finetuning```是用来对1中预训练后的模型，使用```zeroshot/twitter-financial-news-sentiment```进行微调的代码<br>
输出数据为```output/tweets.csv```，用于后续进行可靠性分析
3. ```SMART_Roberta_Large_PhraseBank```是用来对1中预训练后的模型，使用```data/Sentences_AllAgree.csv```进行微调的代码<br>
输出数据为```output/phrase_bank_pred.csv```，实际上是seeking alpha articles数据，用于后续进行可靠性分析
### 命令说明
由于实验环境使用的是CityU提供的htgc1 (https://cslab.cs.cityu.edu.hk/services/high-throughput-gpu-cluster-1-htgc1),需要用脚本进行任务提交
* 以```.condor```为后缀的文件均为任务提交文件，使用以下命令执行：
    ```shell
    condor_submit condor_tweets.condor
    ```
### 环境说明
```RoBERTa-Large-conda-env.yaml```是在htgc1上进行实现的conda虚拟环境配置文件
