# Evaluation模块

该部分代码主要由两部分功能：

## 1 KOLs 可靠性分析
1. Reliability_evaluation_tweets.py 和 Reliability_evaluation_phrasebank.py主要是根据大模型的情感分析输出来进行可靠性分析
2. 可靠性分析结果保存在了output文件夹下, 根据可靠性分析生成的KOLs准确度排名打印在了命令行输出中
3. Ranking_result.txt是进行论文写作时使用的数据汇总

## 2 股价预测
1. 利用大模型情感输出和可靠性分析结果进行股价预测
2. 详细内容请见stock_predict.ipynb文件