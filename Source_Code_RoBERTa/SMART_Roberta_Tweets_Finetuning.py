import pandas as pd
import numpy as np
import math
import sys
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    roc_curve,
    auc,
    average_precision_score,
    f1_score,
)


# check environment
print("==================================================================================")
print('cuda version is ' + str(torch.version.cuda))
print('torch version is ' + torch.__version__)
print('is cuda available? - ' + str(torch.cuda.is_available()))
print('is cudnn enabled? - ' + str(torch.backends.cudnn.enabled))
print(sys.version)
print("==================================================================================")

# load data
train_data = datasets.load_dataset('zeroshot/twitter-financial-news-sentiment', split='train')
test_data = datasets.load_dataset('zeroshot/twitter-financial-news-sentiment', split='validation')

train_data['text']

# PARAMETERS SETTING
model_name = "roberta-large"
device = torch.device("cuda")
tokenizer_name = model_name

num_labels = 3
max_seq_length = 128
train_batch_size = 8
test_batch_size = 8
warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs = 15
learning_rate = 1e-05
adam_epsilon = 1e-08


# 创建一个基于 RoBERTa 的自定义序列分类模型
class SMARTRobertaClassificationModel(nn.Module):

    def __init__(self, model, weight=0.02):
        # a scalar that scales the contribution of the SMART loss, defaulting to 0.02
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):
        # 定义模型的前向传递

        embed = self.model.roberta.embeddings(input_ids)
        # Get initial embeddings
        # Define eval function
        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs[0]
            logits = self.model.classifier(pooled)
            return logits

        # Define SMART loss
        smart_loss_fn = SMARTLoss(eval_fn=eval, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
        # Compute initial (unperturbed) state
        state = eval(embed)
        # Apply classification loss
        loss = F.cross_entropy(state.view(-1, 3), labels.view(-1))
        # Apply smart loss
        loss += self.weight * smart_loss_fn(embed, state)

        return state, loss


# 设置分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
model = (AutoModelForSequenceClassification
         .from_pretrained('./roberta_pretrained_fin_wyc_0', config=config))

# 实例化模型
model_smart = SMARTRobertaClassificationModel(model)
print("==================================================================================")
print('Model=\n',model_smart,'\n')
print("==================================================================================")


class MyClassificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        text, labels = data
        self.examples = tokenizer(text=text, text_pair=None, truncation=True, padding="max_length",
                                  max_length=max_seq_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


train_examples = (train_data['text'], train_data['label'])
train_dataset = MyClassificationDataset(train_examples, tokenizer)

test_examples = (test_data['text'], test_data['label'])
test_dataset = MyClassificationDataset(test_examples, tokenizer)


def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs


train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=train_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset,sampler=test_sampler, batch_size=test_batch_size)

t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
optimizer_grouped_parameters = []
custom_parameter_names = set()
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters.extend(
    [
        {
            "params": [
                p
                for n, p in model_smart.named_parameters()
                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model_smart.named_parameters()
                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
)

warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, no_deprecation_warning=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)


def compute_metrics(preds, model_outputs, labels, eval_examples=None, multi_label=True):
    assert len(preds) == len(labels)
    mismatched = labels != preds

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    con_m = confusion_matrix(labels, preds, labels=[0, 1, 2])

    return (
        {
            **{"mcc": mcc, "acc":acc, "f1": f1},
        },
        con_m
    )

model_smart.to(device)
torch.cuda.empty_cache()
model_smart.zero_grad()

for epoch in range(num_train_epochs):

    model_smart.train()
    epoch_loss = []

    for batch in tqdm(train_dataloader):
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Original Code
        logits, loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        # Free up GPU memory
        torch.cuda.empty_cache()

        scheduler.step()
        model_smart.zero_grad()
        epoch_loss.append(loss.item())

    #    SAVE
    PATH = "SMART_Roberta_large_FinancialTweets/wyc-01/" + str(epoch)
    torch.save(model_smart.state_dict(), PATH)
    print("==================================================================================")
    print('epoch', epoch, 'Training avg loss', np.mean(epoch_loss))


model_smart.to(device)
# fine-tuning
for epoch in range(num_train_epochs):

    # evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dataloader)
    preds = np.empty((len(test_dataset), num_labels))
    out_label_ids = np.empty((len(test_dataset)))
    PATH = "SMART_Roberta_large_FinancialTweets/wyc-01/" + str(epoch)
    model_smart.load_state_dict(torch.load(PATH))
    model_smart.eval()

    for i, test_batch in enumerate(test_dataloader):

        test_batch = get_inputs_dict(test_batch)
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        labels = test_batch['labels'].to(device)
        logits, tmp_eval_loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(test_dataset)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, con_m = compute_metrics(preds, model_outputs, out_label_ids)

    print("==================================================================================")
    print('epoch', epoch, 'Testing  avg loss', eval_loss)
    print(result)
    print(con_m)
    print('---------------------------------------------------\n')
    print("==================================================================================")


df_sample =  pd.read_csv("../data/tweets.csv")
print("=df_sample========================================================================")
print("Column names:")
print(df_sample.columns.tolist())

print("\nFirst 5 rows of data:")
print(df_sample.head())
print("==================================================================================")

new_labels=np.zeros(704)

sample_examples = (df_sample['text'].astype(str).tolist(), new_labels)
sample_dataset = MyClassificationDataset(sample_examples,tokenizer)
sample_dataloader = DataLoader(sample_dataset,shuffle=False,batch_size=test_batch_size)

torch.cuda.empty_cache()
# 评估特定数据集（特别是第 5 轮和第 6 轮）上的预训练模型，并存储第 6 轮的最终预测
model_smart.to(device)
pred_final = []
for epoch in range(5, 7):

    # evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(sample_dataloader)
    preds = np.empty((len(sample_dataset), num_labels))
    out_label_ids = np.empty((len(sample_dataset)))

    PATH = "SMART_Roberta_large_FinancialTweets/wyc-01/" + str(epoch)
    model_smart.load_state_dict(torch.load(PATH))
    model_smart.eval()

    for i, test_batch in enumerate(sample_dataloader):

        test_batch = get_inputs_dict(test_batch)
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        labels = test_batch['labels'].to(device)
        logits, tmp_eval_loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(sample_dataset)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, con_m = compute_metrics(preds, model_outputs, out_label_ids)
    if epoch == 6:
        pred_final = preds


pred_final_df=pd.DataFrame(pred_final)
print("=pred_final_df===================================================================")
print("Column names:")
print(pred_final_df.columns.tolist())

print("\nFirst 5 rows of data:")
print(pred_final_df.head())
print("=================================================================================")

df_sample['pred_sen'] = pred_final_df[0]

torch.cuda.empty_cache()
df_sample.to_csv("./output/tweets.csv", index=False)
