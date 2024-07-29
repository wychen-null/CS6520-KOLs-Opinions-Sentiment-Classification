import pandas as pd
import datasets
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split


# ZERO_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas:
#
# {newsgroup_classes}
#
# From the above list of classes, select only one class that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the class from the given list of classes. Do not predict anything else.
#
# ### Sentence: ```{sentence}```
# ### Class:
# """
#
# FEW_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas:
#
# {newsgroup_classes}
#
# From the above list of classes, select only one class that the provided sentence can be classified into. Once again, only predict the class from the given list of classes. Do not predict anything else. The sentence will be delimited with triple backticks. To help you, examples are provided of sentence and the corresponding class they belong to.
#
# {few_shot_samples}
#
# ### Sentence: ```{sentence}```
# ### Class:
# """
#
# TRAINING_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.
#
# ### Sentence: ```{sentence}```
# ### Class: {label}
# """
#
# INFERENCE_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.
#
# ### Sentence: ```{sentence}```
# ### Class:
# """

TRAINING_CLASSIFIER_PROMPT_v2 = """### This is a sentiment analysis task, using 2 for neutral sentiment, 1 for positive sentiment, and 0 for negative sentiment. Sentence:{sentence} ### Class:{label}"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """### This is a sentiment analysis task, using 2 for neutral sentiment, 1 for positive sentiment, and 0 for negative sentiment. Sentence:{sentence} ### Class:"""

# ZERO_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.
#
# ### Dialogue: ```{dialogue}```
# ### Summary:
# """
#
# FEW_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks. To help you, examples of summarization are provided.
#
# {few_shot_samples}
#
# ### Dialogue: ```{dialogue}```
# ### Summary:
# """
#
# TRAINING_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.
#
# ### Dialogue: ```{dialogue}```
# ### Summary: {summary}
# """
#
# TRAINING_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:{summary}"""
# INFERENCE_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:"""
#
# INFERENCE_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.
#
# ### Dialogue: ```{dialogue}```
# ### Summary:
# """


def get_newsgroup_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text,
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text,
            )
        instructions.append(example)

    return instructions


# def clean_newsgroup_data(texts, labels):
#     label2data = {}
#     clean_data, clean_labels = [], []
#     for data, label in zip(texts, labels):
#         if isinstance(data, str) and isinstance(label, str):
#             clean_data.append(data)
#             clean_labels.append(label)
#
#             if label not in label2data:
#                 label2data[label] = data
#
#     return label2data, clean_data, clean_labels


def get_data_for_ft(mode="train", train_sample_fraction=0.99):
    # 当tweets数据进行读取的时候，使用这部分代码
    train_data = load_dataset('zeroshot/twitter-financial-news-sentiment', split='train')
    train_labels = train_data["label"]
    train_data = train_data["text"]

    test_data = load_dataset('zeroshot/twitter-financial-news-sentiment', split='validation')
    test_labels = test_data["label"]
    test_data = test_data["text"]

    # 当对phrasebank数据进行读取的时候，使用这部分代码
    # dataset = pd.read_csv("./data/Sentences_AllAgree.csv")
    # label_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
    # dataset['Sentiment'] = dataset['Sentiment'].replace(label_mapping)

    # X = dataset["Sentence"]
    # y = dataset["Sentiment"]
    # train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # mix data
    # dataset = pd.read_csv("./data/combined_shuffled.csv")
    # X = dataset["text"]
    # y = dataset["label"]
    # train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # sample n points from training data
    train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})
    train_df, _ = train_test_split(
        train_df,
        train_size=train_sample_fraction,
        stratify=train_df["label"],
        random_state=42,
    )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    # 去除text中的链接，减少对结果的影响
    url_pattern = r'https?://\S+'
    train_data = train_data.str.replace(url_pattern, '', regex=True)

    train_instructions = get_newsgroup_instruction_data(mode, train_data, train_labels)
    test_instructions = get_newsgroup_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    return train_dataset, test_dataset


def get_data_for_output(mode="inference", train_sample_fraction=0.99):

    # tweets 输出结果
    dataset = pd.read_csv("./data/tweets.csv")
    valid_data = dataset['text']
    labels = dataset['pred_sen']
    # seeking alpha输出结果
    # dataset = pd.read_csv("./data/seeking_alpha_pred.csv")
    # valid_data = dataset['summary']
    # labels = dataset['pred_sen']

    url_pattern = r'https?://\S+'
    valid_data = valid_data.str.replace(url_pattern, '', regex=True)

    test_instructions = get_newsgroup_instruction_data(mode, valid_data, labels)

    valid_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": labels,
            }
        )
    )
    return valid_dataset


# def get_newsgroup_data():
#     newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
#     train_data = newsgroup_dataset["train"]["text"]
#     train_labels = newsgroup_dataset["train"]["label"]
#
#     label2data, clean_data, clean_labels = clean_newsgroup_data(
#         train_data, train_labels
#     )
#     df = pd.DataFrame(data={"text": clean_data, "label": clean_labels})
#
#     newsgroup_classes = df["label"].unique()
#     newsgroup_classes = ", ".join(newsgroup_classes)
#
#     few_shot_samples = ""
#     for label, data in label2data.items():
#         sample = f"Sentence: {data} \n Class: {label} \n\n"
#         few_shot_samples += sample
#
#     return newsgroup_classes, few_shot_samples, df
#
#
# def get_samsum_data():
#     samsum_dataset = load_dataset("samsum")
#     train_dataset = samsum_dataset["train"]
#     dialogues = train_dataset["dialogue"][:2]
#     summaries = train_dataset["summary"][:2]
#
#     few_shot_samples = ""
#     for dialogue, summary in zip(dialogues, summaries):
#         sample = f"Sentence: {dialogue} \n Summary: {summary} \n\n"
#         few_shot_samples += sample
#
#     return few_shot_samples
