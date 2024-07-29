import argparse
import torch
import os
import pandas as pd
import evaluate
import pickle
import warnings
from tqdm import tqdm
import csv

from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from transformers import AutoModelForCausalLM
import torch.nn as nn

from prompts import get_data_for_ft

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only the first GPU

# 向隐藏层添加了一个mlp进行降维
class CustomModelWithMLP(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Assuming the model outputs hidden states of size `hidden_size`
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size/2),
            nn.ReLU(),
            nn.Linear(config.hidden_size/2, 3),
            nn.ReLU()
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # Apply MLP to each token's hidden state
        sequence_output = self.mlp(sequence_output)
        return (sequence_output,) + outputs[1:]


def read_folders(directory):
    # List all entries in the given directory
    entries = os.listdir(directory)
    folders = []

    # Iterate over the entries
    for entry in entries:
        full_path = os.path.join(directory, entry)

        # Check if the entry is a directory
        if os.path.isdir(full_path):
            folders.append(full_path)

    return folders


def main(args):
    _, test_dataset = get_data_for_ft(mode="inference")

    # experiment = args.experiment_dir
    folders = read_folders(args.experiment_dir)
    folders.sort()

    for experiment in folders:
        print("The folder is: ", experiment)
        peft_model_id = f"{experiment}/assets"

        # unpatch flash attention
        unplace_flash_attn_with_attn()

        # load base LLM model and tokenizer
        model = CustomModelWithMLP.from_pretrained(
            peft_model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

        results = []
        oom_examples = []
        instructions, labels = test_dataset["instructions"], test_dataset["labels"]

        for instruct, label in tqdm(zip(instructions, labels)):
            input_ids = tokenizer(
                instruct, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            with torch.inference_mode():
                try:
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        do_sample=True,
                        top_p=0.95,
                        temperature=1e-3,
                    )
                    result = tokenizer.batch_decode(
                        outputs.detach().cpu().numpy(), skip_special_tokens=True
                    )[0]
                    result = result[len(instruct):]
                except:
                    result = ""
                    oom_examples.append(input_ids.shape[-1])

                results.append(result)

        labels = [int(label) for label in labels]

        # 使用列表推导式替换非数字元素
        for index in range(len(results)):
            if results[index] not in [0, 1, 2, '0', '1', '2']:
                results[index] = int(2)
            else:
                results[index] = int(results[index])
        # 用于查看输出结果异常值
        # with open('results_mlp.csv', 'w', newline='') as file:
            # writer = csv.writer(file)
            # Write the results to the CSV file
            # writer.writerow(results) 

        metrics = {
            "micro_f1": f1_score(labels, results, average="micro"),
            "macro_f1": f1_score(labels, results, average="macro"),
            "precision": precision_score(labels, results, average="macro"),
            "recall": recall_score(labels, results, average="macro"),
            "accuracy": accuracy_score(labels, results),
            "oom_examples": oom_examples,
        }
        cm = confusion_matrix(labels, results)
        print(cm)

        print(metrics)

        save_dir = os.path.join(experiment, "metrics")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
            pickle.dump(metrics, handle)

        print(f"Completed experiment {peft_model_id}")
        print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
    )

    args = parser.parse_args()
    main(args)
