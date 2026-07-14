import os
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# =====================================================
# Path Configuration
# =====================================================
model_path = "autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

TRAIN_FILE = "train.jsonl"

OUTPUT_DIR = "/root/autodl-tmp/outputs/llama_lora"

# =====================================================
# tokenizer
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Required for Gemma
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# 4-bit Quantization Config (Core)
# =====================================================
bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)

# =====================================================
# model
# =====================================================
model = AutoModelForCausalLM.from_pretrained(

    model_path,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16
)

# =====================================================
# Gemma Training Required Settings
# =====================================================
model.config.use_cache = False

model.gradient_checkpointing_enable()

# =====================================================
# QLoRA Preparation
# =====================================================
model = prepare_model_for_kbit_training(model)

# =====================================================
# LoRA Configuration
# =====================================================
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)

# =====================================================
# Apply LoRA
# =====================================================
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

# =====================================================
# Load Data
# =====================================================
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE
    }
)

# =====================================================
# messages -> text
# =====================================================
def build_chat_text(messages):

    text = ""

    for msg in messages:

        role = msg["role"]
        content = msg["content"]

        text += f"<|{role}|>\n{content}\n"

    return text

# =====================================================
# tokenize
# =====================================================
MAX_LENGTH = 1024

def tokenize_function(example):

    text = build_chat_text(example["messages"])

    tokens = tokenizer(

        text,

        truncation=True,

        max_length=MAX_LENGTH,

        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names
)

# =====================================================
# collator
# =====================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =====================================================
# training args
# =====================================================
training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # batch
    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    # train
    num_train_epochs=2,

    learning_rate=2e-4,

    # logging
    logging_steps=10,

    save_steps=100,

    save_strategy="steps",

    # mixed precision
    fp16=True,

    # optimizer
    optim="paged_adamw_8bit",

    # scheduler
    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    # checkpoint
    save_total_limit=2,

    # report
    report_to="none"
)

# =====================================================
# trainer
# =====================================================
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    data_collator=data_collator
)

# =====================================================
# train
# =====================================================
trainer.train()

# =====================================================
# save
# =====================================================
trainer.save_model(
    os.path.join(OUTPUT_DIR, "final")
)

tokenizer.save_pretrained(
    os.path.join(OUTPUT_DIR, "final")
)

print("\nTraining Finished!")