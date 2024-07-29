from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForPreTraining
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset

import torch
import torch.nn as nn
import os
import psutil


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
torch.cuda.empty_cache()

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used


model_name = "roberta-large"

num_labels = 2
device = torch.device("cuda")
device_ids=[0, 1]

tokenizer_name = model_name

batch_size = 4 # cuda out of memory - reduce it
warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs =100
learning_rate = 1e-05
adam_epsilon = 1e-08

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForPreTraining.from_pretrained(model_name)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.5
)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data/further_training_data/train.txt',
    # file_path='/public/wangyuchen/project/data/tweets/train.txt',
    block_size=512,

)
print(len(train_dataset))

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data/further_training_data/test.txt',
    # file_path='/public/wangyuchen/project/data/tweets/test.txt',
    block_size=512,
)
print(len(eval_dataset))

# t_total = (len(train_dataset)//batch_size+1)//gradient_accumulation_steps*num_train_epochs
# t_total=19890
# print(t_total)
# warmup_steps = math.ceil(t_total * warmup_ratio)
# optimizer = AdamW(model.parameters(),lr=learning_rate, eps=adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)



training_args = TrainingArguments(
    output_dir="./roberta-retrained-wyc-0",
    evaluation_strategy = "epoch",
    save_strategy  = "epoch",
#    warmup_ratio = warmup_ratio,
    learning_rate = learning_rate,
    overwrite_output_dir=True,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #eval_steps=400,
    #save_steps=400,
    save_total_limit=3,
    seed=1,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,

    fp16=True, # Enable mixed precision training
    gradient_accumulation_steps=4, # Use gradient accumulation
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
)

model= nn.DataParallel(model)
#model.to(device)
print("==START TRAINING======================================================")
gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
          .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
print(r'当前机器内存使用情况：总共 {} MB， 剩余 {} MB, 当前进程使用的内存 {} MB'
          .format(cpu_mem_total, cpu_mem_free, cpu_mem_process_used))
trainer.train()


model.module.save_pretrained("./roberta_pretrained_fin_wyc_0")