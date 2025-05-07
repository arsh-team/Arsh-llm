# Copyright 2025 Arshia Afshani & Arsh AI team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
from typing import Optional, Dict, Union, List
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PhiTokenizer
)
from datasets import load_dataset


# ==================== Optimized Tokenizer Configuration ====================
class ArshAITokenizer(PhiTokenizer):
    """Enhanced Phi-4 tokenizer with extended special tokens and chat capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extended token configuration
        self.model_max_length = 16384
        self.chat_template = self._build_chat_template()
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add domain-specific special tokens with precise attributes"""
        new_tokens = [
                         ("<|dummy_{}|>".format(i), 100256 + i) for i in range(88)
                     ] + [
                         ("<|fim_prefix|>", 100258),
                         ("<|fim_middle|>", 100259),
                         ("<|fim_suffix|>", 100260),
                         ("<|im_start|>", 100264),
                         ("<|im_end|>", 100265),
                         ("<|im_sep|>", 100266)
                     ]

        for token, token_id in new_tokens:
            self.add_tokens([token], special_tokens=True)
            self.added_tokens_decoder[token_id] = token

    def _build_chat_template(self):
        """Optimized chat template with efficient string operations"""
        return (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}<|im_sep|>{{ message.content }}<|im_end|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant<|im_sep|>{% endif %}"
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Enhanced save with full configuration"""
        super().save_pretrained(save_directory, **kwargs)

        config = {
            "model_max_length": self.model_max_length,
            "chat_template": self.chat_template,
            "added_tokens": list(self.added_tokens_decoder.values()),
            "padding_side": "left",
            "truncation_side": "right"
        }

        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)


# ==================== Phi-4 Enhanced Model Architecture ====================
# ==================== Complete Phi-4 Weight Mapping ====================
def get_phi4_mapping():
    """Returns complete weight mapping between Phi-4 and custom architecture"""
    return {
        # Embeddings
        "transformer.embd.wte.weight": "model.embed_tokens.weight",

        # Final LayerNorm
        "transformer.ln_f.weight": "model.norm.weight",

        # LM Head
        "lm_head.weight": "lm_head.weight",

        # Layer-wise mappings (applied to all layers)
        "transformer.h.{}.ln.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.mlp.fc1.weight": "model.layers.{}.mlp.gate_up_proj.weight",
        "transformer.h.{}.mlp.fc2.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.h.{}.mixer.Wqkv.weight": "model.layers.{}.self_attn.qkv_proj.weight",
        "transformer.h.{}.mixer.out_proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.ln_mlp.weight": "model.layers.{}.post_attention_layernorm.weight",

        # Rotary embeddings (if using custom implementation)
        "transformer.h.{}.mixer.rotary_emb.inv_freq": "model.layers.{}.self_attn.rotary_emb.inv_freq",
    }


def load_pretrained_weights(model, phi4_path: str):
    """Complete weight loading with architecture alignment"""
    from safetensors import safe_open

    # Load Phi-4 weights
    phi4_state = {}
    with safe_open(phi4_path, framework="pt") as f:
        for key in f.keys():
            phi4_state[key] = f.get_tensor(key)

    # Get complete mapping
    mapping = get_phi4_mapping()

    # Initialize model state dict
    model_state = model.state_dict()

    # Process each parameter
    for model_key, param in model_state.items():
        # Check direct mappings first
        matched = False

        # Handle layer-specific patterns
        for phi_pattern, model_pattern in mapping.items():
            if "{}" in phi_pattern:
                # Handle layer-wise parameters
                if "layers" in model_key:
                    layer_num = model_key.split(".")[2]
                    phi_key = phi_pattern.format(layer_num)

                    if phi_key in phi4_state:
                        param.copy_(phi4_state[phi_key])
                        matched = True
                        break

            # Handle direct mappings
            elif model_pattern == model_key and phi_pattern in phi4_state:
                param.copy_(phi4_state[phi_pattern])
                matched = True
                break

        # Handle special cases
        if not matched:
            if "lm_head" in model_key and "lm_head.weight" in phi4_state:
                param.copy_(phi4_state["lm_head.weight"])
            elif "embed_tokens" in model_key and "transformer.embd.wte.weight" in phi4_state:
                param.copy_(phi4_state["transformer.embd.wte.weight"])
            elif "norm.weight" in model_key and "transformer.ln_f.weight" in phi4_state:
                param.copy_(phi4_state["transformer.ln_f.weight"])
            else:
                print(f"Warning: No matching parameter found for {model_key}")

    return model


def create_model(phi4_path: str):
    """Create model with Phi-4 optimized architecture"""
    config = AutoConfig.from_pretrained(phi4_path, trust_remote_code=True)

    # Enhanced configuration for reasoning
    config.update({
        "hidden_size": 5120,
        "intermediate_size": 17920,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "max_position_embeddings": 16384,
        "rope_theta": 250000,
        "rms_norm_eps": 1e-5,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2"
    })

    model = AutoModelForCausalLM.from_config(config)
    model = load_pretrained_weights(model, phi4_path)
    return model


# ==================== TPU-Optimized Training Loop ====================
def train_fn(index):
    """Distributed training function with XLA optimizations"""
    device = xm.xla_device()

    # Stream dataset for memory efficiency
    dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
    tokenizer = ArshAITokenizer.from_pretrained("./arsh-tokenizer")

    # Dynamic batch processing
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=128  # TPU alignment
    )

    model = create_model("microsoft/phi-4").to(device)

    # Sophia optimizer configuration
    training_args = TrainingArguments(
        output_dir="./arsh-ai",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.1,
        bf16=True,
        logging_steps=100,
        save_strategy="steps",
        save_steps=5000,
        xla=True,
        report_to="tensorboard",
        optim="sophia",
        gradient_checkpointing=True,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    xm.save(model.state_dict(), f"arsh-ai-model-{index}.pt")


# ==================== Execution Setup ====================
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = ArshAITokenizer.from_pretrained("microsoft/phi-4")
    tokenizer.save_pretrained("./arsh-tokenizer")

    # Launch TPU training
    xmp.spawn(train_fn, args=(), nprocs=8)
