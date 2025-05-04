import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import (
    GPT2Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from transformers import LlamaTokenizer
from typing import Dict, Optional, Union, List
import json


# ==================== تنظیمات توکنایزر اختصاصی ====================
class ArshLlamaTokenizer(LlamaTokenizer):
    def __init__(
            self,
            vocab_file: str,
            merges_file: str,
            tokenizer_file: Optional[str] = None,
            unk_token: Union[str, Dict] = "ï¿½",
            bos_token: Union[str, Dict] = "<|endoftext|>",
            eos_token: Union[str, Dict] = "<|im_end|>",
            pad_token: Union[str, Dict] = "<|dummy_87|>",
            add_bos_token: bool = False,
            add_prefix_space: bool = False,
            **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_prefix_space=add_prefix_space,
            **kwargs
        )

        # تنظیمات خاص از کانفیگ
        self.model_max_length = 16384
        self.legacy = False
        self.errors = "replace"
        self.clean_up_tokenization_spaces = False
        self.padding_side = "left"
        self.truncation_side = "right"
        self.stride = 0
        self.max_length = 2048

        # افزودن توکن‌های خاص با ویژگی‌های دقیق
        special_tokens = [
            {"id": 5809, "content": "ï¿½", "lstrip": False, "rstrip": False, "normalized": False, "single_word": False},
            {"id": 100257, "content": "<|endoftext|>", "lstrip": True, "rstrip": True, "normalized": False,
             "single_word": False},
            # ... سایر توکن‌ها با همان ساختار
        ]

        # تولید خودکار توکن‌های dummy
        dummy_tokens = [{"id": 100256 + i, "content": f"<|dummy_{i}|>", "lstrip": True, "rstrip": True} for i in
                        range(88)]
        fim_tokens = [
            {"id": 100258, "content": "<|fim_prefix|>", "lstrip": True, "rstrip": True},
            {"id": 100259, "content": "<|fim_middle|>", "lstrip": True, "rstrip": True},
            {"id": 100260, "content": "<|fim_suffix|>", "lstrip": True, "rstrip": True}
        ]
        im_tokens = [
            {"id": 100264, "content": "<|im_start|>", "lstrip": True, "rstrip": True},
            {"id": 100265, "content": "<|im_end|>", "lstrip": True, "rstrip": True},
            {"id": 100266, "content": "<|im_sep|>", "lstrip": True, "rstrip": True}
        ]

        all_special_tokens = special_tokens + dummy_tokens + fim_tokens + im_tokens

        for token in all_special_tokens:
            self.add_tokens(
                [token["content"]],
                special_tokens=True,
                lstrip=token.get("lstrip", False),
                rstrip=token.get("rstrip", False),
                normalized=token.get("normalized", False),
                single_word=token.get("single_word", False)
            )

        # تنظیم شناسه‌های خاص
        self.bos_token_id = 100257
        self.eos_token_id = 100265
        self.pad_token_id = 100351
        self.unk_token_id = 5809

    @property
    def chat_template(self):
        return """{% for message in messages %}
            {% if message['role'] == 'system' %}
                {{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}
            {% elif message['role'] == 'user' %}
                {{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}
            {% elif message['role'] == 'assistant' %}
                {{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}
            {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"""

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)

        # ذخیره کانفیگ دقیق
        config = {
            "add_bos_token": self.add_bos_token,
            "add_prefix_space": self.add_prefix_space,
            "added_tokens_decoder": {str(k): v for k, v in self.added_tokens_decoder.items()},
            "bos_token": self.bos_token,
            "chat_template": self.chat_template,
            "clean_up_tokenization_spaces": self.clean_up_tokenization_spaces,
            "eos_token": self.eos_token,
            "errors": self.errors,
            "legacy": self.legacy,
            "model_max_length": self.model_max_length,
            "pad_token": self.pad_token,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "unk_token": self.unk_token
        }

        with open(f"{save_directory}/tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=4)

    @property
    def chat_template(self):
        return """{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"""


# ==================== تنظیمات مدل ====================
def get_model():
    config = LlamaConfig(
        hidden_size=5120,
        intermediate_size=17920,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=10,
        head_dim=128,
        rope_theta=250000,
        max_position_embeddings=16384,
        vocab_size=100352,
        rms_norm_eps=1e-5,
        pretraining_tp=1,
        torch_dtype=torch.float16,
    )

    model = LlamaForCausalLM(config)

    # Phi-style وزن دهی اولیه با استفاده از
    def phi_initialization(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(phi_initialization)
    return model


# ==================== پردازش داده ====================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=16384,
        padding="max_length",
        return_tensors="pt",
    )


# ==================== آموزش توزیع شده روی TPU ====================
def train_fn(index):
    # تنظیم دستگاه
    device = xm.xla_device()

    # بارگیری داده‌ها
    dataset = load_dataset("EleutherAI/pile", split="train")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # تنظیم مدل
    model = get_model().to(device)

    # آرگومان‌های آموزش
    training_args = TrainingArguments(
        output_dir="./arsh-pretrained",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10000,
        logging_steps=500,
        learning_rate=3e-4,
        weight_decay=0.01,
        fp16=True,
        xla_backend="XLA",
        dataloader_drop_last=True,
    )

    # Trainer تنظیم
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # شروع آموزش
    trainer.train()
    xm.save(model.state_dict(), f"arsh_model_final_{index}.pt")


# ==================== اجرای اصلی ====================
if __name__ == "__main__":
    # مقداردهی توکنایزر
    tokenizer = ArshLlamaTokenizer.from_pretrained("microsoft/phi-4")
    tokenizer.save_pretrained("./arsh-tokenizer")

    # شروع آموزش روی TPU
    xmp.spawn(train_fn, args=())
