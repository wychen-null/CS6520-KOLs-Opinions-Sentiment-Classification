import pandas as pd
import numpy as np
import math
import sys
import torch.utils.data as data
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    random_split
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

all_df = pd.read_csv('../data/FinancialPhraseBank-v1.0/Sentences_AllAgree.csv')
label = []
for index, row in all_df.iterrows():
    if row['Sentiment'] == 'negative':
        label.append(0)
    elif row['Sentiment'] == 'neutral':
        label.append(2)
    else:
        label.append(1)

label_df = pd.DataFrame(label)
label_df.columns = ['Class']
all_df = all_df.join(label_df)
print(all_df.head)

train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)
# Save the training DataFrame to a CSV file
train_df.to_csv('train_dataset.csv', index=False)

# Save the testing DataFrame to a CSV file
test_df.to_csv('test_dataset.csv', index=False)


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


class SMARTRobertaClassificationModel(nn.Module):

    def __init__(self, model, weight=0.02):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):
        # Get initial embeddings
        embed = self.model.roberta.embeddings(input_ids)

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


tokenizer = AutoTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, config=config)

model_smart = SMARTRobertaClassificationModel(model)
print("==================================================================================")
print('Model=\n',model_smart,'\n')
print("==================================================================================")

def subset_to_dataframe(subset):
    # Initialize lists to hold data
    sentences = []
    classes = []

    # Assuming each item in the subset is a tuple (sentence, class)
    for item in subset:
        sentences.append(item[0])
        classes.append(item[1])

    # Create a DataFrame
    df = pd.DataFrame({
        'Sentence': sentences,
        'Class': classes
    })
    return df

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


train_examples = (train_df['Sentence'].astype(str).tolist(), train_df['Class'].tolist())
train_dataset = MyClassificationDataset(train_examples, tokenizer)

test_examples = (test_df['Sentence'].astype(str).tolist(), test_df['Class'].tolist())
test_dataset = MyClassificationDataset(test_examples, tokenizer)


def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=train_batch_size)

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
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

def compute_metrics(preds, model_outputs, labels, eval_examples=None, multi_label=True):
    assert len(preds) == len(labels)
    mismatched = labels != preds
    #wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    con_m = confusion_matrix(labels, preds, labels=[0, 1, 2])
#     scores = np.array([softmax(element)[1] for element in model_outputs])
#     fpr, tpr, thresholds = roc_curve(labels, scores)
#     auroc = auc(fpr, tpr)
#     auprc = average_precision_score(labels, scores)
    return (
        {
            **{"mcc": mcc, "acc":acc, "f1": f1},
        },
        con_m
    )

def print_confusion_matrix(result):
    print('confusion matrix:')
    print('            predicted    ')
    print('          0     |     1')
    print('    ----------------------')
    print('   0 | ',format(result['tn'],'5d'),' | ',format(result['fp'],'5d'))
    print('gt -----------------------')
    print('   1 | ',format(result['fn'],'5d'),' | ',format(result['tp'],'5d'))
    print('---------------------------------------------------')

torch.cuda.empty_cache()

model_smart.to(device)
model_smart.zero_grad()

# training
for epoch in range(num_train_epochs):

    model_smart.train()
    epoch_loss = []

    for batch in tqdm(train_dataloader):
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits, loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model_smart.zero_grad()
        epoch_loss.append(loss.item())

    #    SAVE
    PATH = "SMART_Roberta_large_FinancialPhraseBank/all_agree/" + str(epoch)
    torch.save(model_smart.state_dict(), PATH)

    print('epoch', epoch, 'Training avg loss', np.mean(epoch_loss))

model_smart.to(device)

for epoch in range(num_train_epochs):

    # evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dataloader)
    preds = np.empty((len(test_dataset), num_labels))
    out_label_ids = np.empty((len(test_dataset)))

    PATH = "SMART_Roberta_large_FinancialPhraseBank/all_agree/" + str(epoch)
    model_smart.load_state_dict(torch.load(PATH))
    model_smart.eval()

    for i, test_batch in enumerate(test_dataloader):
        #         with torch.no_grad():
        test_batch = get_inputs_dict(test_batch)
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        labels = test_batch['labels'].to(device)
        logits, tmp_eval_loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        #             tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(test_dataset)
        #         print(logits)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, con_m = compute_metrics(preds, model_outputs, out_label_ids)

    # print('epoch',epoch,'Training avg loss',np.mean(epoch_loss))
    print('epoch', epoch, 'Testing  avg loss', eval_loss)
    print(result)
    print(con_m)
    print('---------------------------------------------------\n')


df_sample =  pd.read_csv("../data/seeking_alpha_cleaned.csv")
# df_sample
new_labels=np.zeros(234)

sample_examples = (df_sample['summary'].astype(str).tolist(),new_labels)
sample_dataset = MyClassificationDataset(sample_examples,tokenizer)

sample_dataloader = DataLoader(sample_dataset,shuffle=False,batch_size=test_batch_size)

print(sample_dataloader)

# sample_data = datasets.load_dataset('zeroshot/twitter-financial-news-sentiment', split='validation')

# new_label = []
# for s in sample_data:
    # if s['label'] == 1:
#         print('f')
        # new_label.append(2)
    # elif s['label']==2:
        # new_label.append(1)
    # else:
        # new_label.append(0)
# sample_examples = (sample_data['text'], new_label)
# sample_dataset = MyClassificationDataset(sample_examples,tokenizer)
# sample_dataloader = DataLoader(sample_dataset,shuffle=False,batch_size=test_batch_size)

model_smart.to(device)
pred_final = []
for epoch in range(num_train_epochs):

    # evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(sample_dataloader)
    preds = np.empty((len(sample_dataset), num_labels))
    out_label_ids = np.empty((len(sample_dataset)))

    PATH = "SMART_Roberta_large_FinancialPhraseBank/all_agree/" + str(epoch)
    model_smart.load_state_dict(torch.load(PATH))
    model_smart.eval()

    for i, test_batch in enumerate(sample_dataloader):
        #         with torch.no_grad():
        test_batch = get_inputs_dict(test_batch)
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        labels = test_batch['labels'].to(device)
        logits, tmp_eval_loss = model_smart(input_ids, attention_mask=attention_mask, labels=labels)
        #             tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(sample_dataset)
        #         print(logits)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, con_m = compute_metrics(preds, model_outputs, out_label_ids)
    if epoch == 8:
        pred_final = preds

        # print('epoch',epoch,'Training avg loss',np.mean(epoch_loss))
    print('epoch', epoch, 'Testing  avg loss', eval_loss)
    print(result)
    print(con_m)
    print('---------------------------------------------------\n')


pred_final_df=pd.DataFrame(preds)
pred_final_df.columns = ['pred_sen']
df_sample = df_sample.join(pred_final_df)

df_sample.to_csv("./output/phrase_bank_pred.csv", index=False)
print("##############################Finish#############################################")
