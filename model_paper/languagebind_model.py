"""
LanguageBind_Video_FT (Zhu et al., ICLR 2024) in plain PyTorch.

The official package (github.com/PKU-YuanGroup/LanguageBind) pins
transformers==4.30 and imports CLIP internals that no longer exist in
transformers 5, so the model is re-implemented here and loads the official
weights unchanged (LanguageBind/LanguageBind_Video_FT, pytorch_model.bin).

Architecture (config.json of the checkpoint):
    video : CLIP ViT-L/14, 8 frames, 224 px. Every encoder layer first runs a
            temporal attention over the 8 frames (for each token position,
            with a learned temporal embedding), then the usual spatial
            attention + MLP. Video embedding = mean over frames of the
            post-LayerNorm CLS token, projected to 768-D.
    text  : CLIP text transformer (12 layers, 768 wide, causal), pooled at
            the first <|endoftext|> token, projected to 768-D.
    score : cosine(text, video).

Pre-processing as in the official 'opencv' backend: 8 frames, short side
resized to 224 (bilinear), centre crop 224, OpenAI CLIP mean/std.

LoRA (add_lora) is used for fine-tuning on a 6 GB GPU; save() merges the
LoRA deltas, so a fine-tuned folder loads exactly like the base model.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

BASE_MODEL = "LanguageBind/LanguageBind_Video_FT"
NUM_FRAMES = 8
IMAGE_SIZE = 224
MAX_TEXT_LEN = 77
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

WEIGHTS_FILE = "pytorch_model.bin"
COPY_FILES = ("config.json", "vocab.json", "merges.txt", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json")


# ---------------------------------------------------------------------
# Building blocks (parameter names follow the HF checkpoint)
# ---------------------------------------------------------------------
def activation(name):
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads

        def split(t):
            return t.view(b, n, h, d // h).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dim, hidden, act):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = activation(act)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg, num_frames=0):
        super().__init__()
        d, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.self_attn = Attention(d, cfg["num_attention_heads"])
        self.layer_norm1 = nn.LayerNorm(d, eps=eps)
        self.mlp = MLP(d, cfg["intermediate_size"], cfg["hidden_act"])
        self.layer_norm2 = nn.LayerNorm(d, eps=eps)

        self.t = num_frames
        if num_frames:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, d))
            self.temporal_attn = Attention(d, cfg["num_attention_heads"])
            self.temporal_layer_norm1 = nn.LayerNorm(d, eps=eps)

    def forward(self, x, mask=None):
        if self.t:
            # x: (b t) n d. Temporal attention over the t frames, per token.
            t, n = self.t, x.shape[1]
            x = x.view(-1, t, n, x.shape[-1]).transpose(1, 2)        # b n t d
            x = x + self.temporal_embedding[:, None, :t]
            y = self.temporal_attn(self.temporal_layer_norm1(x.flatten(0, 1)))
            x = x + y.view(x.shape)
            x = x.transpose(1, 2).flatten(0, 1)                       # (b t) n d

        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, cfg, num_frames=0):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(cfg, num_frames) for _ in range(cfg["num_hidden_layers"])
        )
        self.gradient_checkpointing = False

    def forward(self, x, mask=None):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return x


class VisionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, p = cfg["hidden_size"], cfg["patch_size"]
        self.class_embedding = nn.Parameter(torch.zeros(d))
        self.patch_embedding = nn.Conv2d(3, d, kernel_size=p, stride=p, bias=False)
        self.position_embedding = nn.Embedding((cfg["image_size"] // p) ** 2 + 1, d)

    def forward(self, pixels):
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1) + self.position_embedding.weight.to(x.dtype)


class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.num_frames = cfg["num_frames"]
        self.embeddings = VisionEmbeddings(cfg)
        self.pre_layrnorm = nn.LayerNorm(d, eps=eps)  # (sic) checkpoint name
        self.encoder = Encoder(cfg, num_frames=cfg["num_frames"] if cfg.get("add_time_attn") else 0)
        self.post_layernorm = nn.LayerNorm(d, eps=eps)

    def forward(self, pixels):
        """
        pixels (B, T, 3, H, W), normalised, T = num_frames -> (B, D).
        """
        b, t = pixels.shape[:2]
        if t != self.num_frames:
            raise ValueError(f"LanguageBind expects {self.num_frames} frames, got {t}.")
        x = self.pre_layrnorm(self.embeddings(pixels.flatten(0, 1)))
        x = self.encoder(x)
        cls = self.post_layernorm(x[:, 0])
        return cls.view(b, t, -1).mean(dim=1)


class TextEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.token_embedding = nn.Embedding(cfg["vocab_size"], d)
        self.position_embedding = nn.Embedding(cfg["max_position_embeddings"], d)

    def forward(self, input_ids):
        n = input_ids.shape[1]
        return self.token_embedding(input_ids) + self.position_embedding.weight[:n]


class TextTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = TextEmbeddings(cfg)
        self.encoder = Encoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def forward(self, input_ids, attention_mask=None):
        b, n = input_ids.shape
        x = self.embeddings(input_ids)

        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None]
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        x = self.final_layer_norm(self.encoder(x, mask))
        # pooled at the first <|endoftext|> (the largest token id)
        eot = input_ids.to(torch.int).argmax(dim=-1)
        return x[torch.arange(b, device=x.device), eot]


# ---------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def delta(self):
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LanguageBindVideo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        v, t = config["vision_config"], config["text_config"]
        self.vision_model = VisionTransformer(v)
        self.text_model = TextTransformer(t)
        self.visual_projection = nn.Linear(v["hidden_size"], config["projection_dim"], bias=False)
        self.text_projection = nn.Linear(t["hidden_size"], config["projection_dim"], bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.get("logit_scale_init_value", 2.6592)))

    # -- features -----------------------------------------------------
    def get_video_features(self, pixels):
        return self.visual_projection(self.vision_model(pixels))

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_projection(self.text_model(input_ids, attention_mask))

    # -- training helpers ---------------------------------------------
    def gradient_checkpointing_enable(self):
        self.vision_model.encoder.gradient_checkpointing = True
        self.text_model.encoder.gradient_checkpointing = True

    def add_lora(self, r=16, alpha=16, dropout=0.1, targets=LORA_TARGETS):
        """
        Freeze every weight and wrap the attention projections of both
        towers (spatial + temporal) with LoRA adapters.
        """
        for p in self.parameters():
            p.requires_grad = False

        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    lora = LoRALinear(child, r, alpha, dropout)
                    lora.to(device=child.weight.device)
                    setattr(module, name, lora)

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merged_state_dict(self):
        """
        State dict with checkpoint names; LoRA deltas folded into the weights.
        """
        out = {}
        for name, tensor in self.state_dict().items():
            if ".lora_" in name:
                continue
            out[name.replace(".base.", ".")] = tensor.detach().clone()

        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                key = f"{name}.weight"
                out[key] = out[key] + module.delta().detach().to(out[key].dtype)

        return {k: v.cpu() for k, v in out.items()}


# ---------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------
def resolve(model_name_or_path):
    path = Path(model_name_or_path)
    if path.is_dir():
        return path

    from huggingface_hub import hf_hub_download

    for f in COPY_FILES + (WEIGHTS_FILE,):
        local = hf_hub_download(str(model_name_or_path), f)
    return Path(local).parent


def load_model(model_name_or_path=BASE_MODEL, device="cpu", dtype=torch.float32):
    folder = resolve(model_name_or_path)

    with open(folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model = LanguageBindVideo(config)
    state = torch.load(folder / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.endswith("position_ids")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Weight mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")

    return model.to(device=device, dtype=dtype).eval(), folder


def load_tokenizer(model_name_or_path=BASE_MODEL):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(str(resolve(model_name_or_path)))


def save_model(model, base_folder, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.merged_state_dict(), output_dir / WEIGHTS_FILE)
    for f in COPY_FILES:
        if (Path(base_folder) / f).exists():
            shutil.copy(Path(base_folder) / f, output_dir / f)


# ---------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------
def frames_to_uint8(frames, size=IMAGE_SIZE):
    """
    list of HxWx3 RGB uint8 frames -> uint8 tensor (T, 3, size, size):
    short side resized to size (bilinear), centre crop.
    """
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = (nh - size) // 2, (nw - size) // 2
    x = x[..., top:top + size, left:left + size]
    return x.round().clamp(0, 255).to(torch.uint8)


def normalize_pixels(x_uint8, device, dtype):
    """
    uint8 (..., 3, H, W) -> normalised float on device.
    """
    x = x_uint8.to(device).float().div_(255.0)
    shape = [1] * (x.dim() - 3) + [3, 1, 1]
    mean = torch.tensor(MEAN, device=device).view(shape)
    std = torch.tensor(STD, device=device).view(shape)
    return ((x - mean) / std).to(dtype)


def tokenize(tokenizer, texts, device, max_length=MAX_TEXT_LEN):
    tok = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {"input_ids": tok["input_ids"].to(device), "attention_mask": tok["attention_mask"].to(device)}
