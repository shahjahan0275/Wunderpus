# Ablation study for NDSS Submission 2026 
# Objective improving visual image quality
# Method :To fix the visual quality without sacrificing attack strength, 
# we must bring the lpips Perceptual Loss and a Total Variation (TV) smoothness penalty 
# directly into Phase 2. By defining this as a new objective for MIG-COW, 
# the algorithm will mathematically constrain the adversarial noise to hide 
# within natural facial textures rather than forming blocky grids.
# Added New loss function loss_fn_perceptual_smoothness and 
# integrated it into the compute_cow_gradient aggregation 

'''
import argparse
import math
import os
import sys
import types
import torch
from torch import optim
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models 
import lpips
import random
import numpy as np
import clip
import pandas as pd
import torch_dct as dct
import traceback
import open_clip

# Added BLIP imports
from transformers import BlipProcessor, BlipForConditionalGeneration

E4E_PATH = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/encoder4editing/e4e_ffhq_encode.pt"

# ==================================================================================
# ---- PATCH SHUFFLING UTILITY FOR LOCAL FEATURE EXTRACTION ----
# ==================================================================================
def shuffle_patches_tensor(tensor_in, grid=4):
    """
    Divides the input image tensor into a grid x grid patches and shuffles them spatially.
    Forces the feature extractor to evaluate local texture/semantics without global structure.
    """
    B, C, H, W = tensor_in.shape
    ph, pw = H // grid, W // grid
    H_new, W_new = ph * grid, pw * grid
    
    # Crop to ensure perfect divisibility
    tensor_cropped = tensor_in[:, :, :H_new, :W_new]
    
    # Extract patches
    patches = tensor_cropped.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.contiguous().view(B, C, grid * grid, ph, pw)
    
    # Shuffle patches
    shuffled = torch.zeros_like(patches)
    for b in range(B):
        idx = torch.randperm(grid * grid, device=tensor_in.device)
        shuffled[b] = patches[b, :, idx, :, :]
        
    # Reassemble the image from shuffled patches
    shuffled = shuffled.view(B, C, grid, grid, ph, pw)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_new, W_new)
    
    if H_new != H or W_new != W:
        shuffled = F.pad(shuffled, (0, W - W_new, 0, H - H_new))
        
    return shuffled

# ==================================================================================
# ---- Transformer FUSION ATTACK (ARCHITECTURAL SURROGATE) ----
# ==================================================================================
class TransformerModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=32, dropout_rate=0.1, num_classes=2):
        super(TransformerModel, self).__init__()
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, y, z):
        x, attn_weights = self.transformer_block(x, y, z)
        x = x.permute(0, 2, 1)  
        x = self.global_avg_pool(x).squeeze(-1) 
        x = self.dropout(x)
        return self.classifier(x), attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim),
                                 nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, y, z):
        attn_output, attn_weights = self.attention(x, y, z)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output)), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim // num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)

    def forward(self, x, y, z, mask=None):
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 0, 2)
        attn_output, attn_weights = self.multihead_attn(x, y, z, attn_mask=mask)
        return attn_output.permute(1, 0, 2), attn_weights  

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.position_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.position_emb(positions)
        return x

def standard_scale(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

class DetectorNet(nn.Module):
    def __init__(self, CLIP_model, TransformerModelClass):
        super(DetectorNet, self).__init__()
        self.TransformerModel = TransformerModelClass(embed_dim=(768), num_heads=8)
        self.CLIP_model = CLIP_model
        self.DCT_Embedder = nn.Linear((320*320), 768, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, Images, Text_Encodings, DCT_features):
        img_embedding = self.CLIP_model.encode_image(Images)
        text_embedding = self.CLIP_model.encode_text(Text_Encodings)
        
        DCT_features_reshaped = DCT_features.reshape(DCT_features.size(0), -1)
        DCT_features_reshaped = torch.log(torch.abs(DCT_features_reshaped) + 1e-12)
        DCT_embedding = standard_scale(DCT_features_reshaped)
        
        DCT_embedding = self.relu(self.DCT_Embedder(DCT_embedding))
        
        combined_embedding = torch.stack([img_embedding, DCT_embedding, text_embedding], dim=1)        
        CrossAttention_out, _ = self.TransformerModel(combined_embedding, combined_embedding, combined_embedding)
        
        return CrossAttention_out

# ==================================================================================
# ---- 1. MIG-COW ALGORITHMS ----
# ==================================================================================
def integrated_gradients(x, baseline, loss_fn, steps=7):
    if torch.allclose(x, baseline):
        x_temp = x.detach().requires_grad_(True)
        loss = loss_fn(x_temp)
        return torch.autograd.grad(loss, x_temp)[0].detach()

    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]

    total_gradients = torch.zeros_like(x)
    for scaled_x in scaled_inputs:
        scaled_x = scaled_x.detach().requires_grad_(True)
        loss = loss_fn(scaled_x)
        grad = torch.autograd.grad(loss, scaled_x)[0]
        total_gradients += grad

    avg_grad = total_gradients / steps
    return avg_grad.detach()

def compute_cow_gradient(grads, beta=0.75, delta=1e-8):
    N = len(grads)
    norm_grads = [g / (g.norm(p=2) + 1e-8) for g in grads]
    g_con = sum(norm_grads) / N

    flattened_grads = [g.view(-1) for g in norm_grads]
    G = torch.stack(flattened_grads, dim=1)
    K = torch.matmul(G.t(), G)
    K = K + torch.eye(K.size(0), device=K.device) * 1e-6

    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    v_min = eigenvectors[:, 0]

    g_agg = torch.zeros_like(grads[0])
    for i in range(N):
        g_agg += v_min[i] * norm_grads[i]

    dot_product = torch.sum(g_agg * g_con)
    norm_sq = torch.sum(g_con * g_con) + delta
    g_orth = g_agg - (dot_product / norm_sq) * g_con

    return beta * g_con + (1 - beta) * g_orth

# ==================================================================================
# ---- 2. STYLEGAN2 PATCH ----
# ==================================================================================
def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        return F.leaky_relu(input + bias.view(1, -1, 1, 1), negative_slope=negative_slope) * scale
    else:
        return F.leaky_relu(input, negative_slope=negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, (list, tuple)): up = (up, up)
    if not isinstance(down, (list, tuple)): down = (down, down)
    if not isinstance(pad, (list, tuple)): pad = (pad, pad)
    n, c, h, w = input.shape
    k_h, k_w = kernel.shape
    kernel = torch.flip(kernel, [0, 1]).view(1, 1, k_h, k_w).repeat(c, 1, 1, 1)
    if up[0] > 1 or up[1] > 1:
        input = input.view(n, c, h, 1, w, 1)
        input = F.pad(input, [0, up[0] - 1, 0, 0, 0, up[1] - 1])
        input = input.view(n, c, h * up[1], w * up[0])
    input = F.pad(input, [max(pad[0], 0), max(pad[1], 0), max(pad[0], 0), max(pad[1], 0)])
    out = F.conv2d(input, kernel, groups=c)
    out = out[:, :, ::down[1], ::down[0]]
    return out

class UpFirDn2d(nn.Module):
    def __init__(self, kernel, up=1, down=1, pad=(0, 0)):
        super().__init__()
        self.register_buffer('kernel', kernel)
        self.up = up
        self.down = down
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, self.up, self.down, self.pad)

op_module = types.ModuleType("op")
op_module.fused_leaky_relu = fused_leaky_relu
op_module.FusedLeakyReLU = FusedLeakyReLU
op_module.upfirdn2d = upfirdn2d
op_module.UpFirDn2d = UpFirDn2d
op_module.conv2d_gradfix = types.ModuleType("conv2d_gradfix")
op_module.conv2d_gradfix.conv2d = torch.nn.functional.conv2d
op_module.conv2d_gradfix.conv_transpose2d = torch.nn.functional.conv_transpose2d
#sys.modules["op"] = op_module
#os.environ['STYLEGAN2_NO_CUSTOM_OPS'] = '1'
# --- UPDATED: Intercept ALL possible import paths for StyleGAN2 JIT ---
sys.modules["op"] = op_module
sys.modules["models.stylegan2.op"] = op_module
sys.modules["models.stylegan2.op.fused_act"] = op_module
sys.modules["models.stylegan2.op.upfirdn2d"] = op_module
# ----------------------------------------------------------------------

os.environ['STYLEGAN2_NO_CUSTOM_OPS'] = '1'

# ==================================================================================
# ---- MAIN SETUP & SCRIPT ----
# ==================================================================================
PROJECT_ROOT = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "encoder4editing"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "encoder4editing", "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "stylegan2-pytorch"))
sys.path.insert(0, "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def generate_wplus():
    STYLEGAN_DIR = os.path.join(PROJECT_ROOT, "stylegan2-pytorch")
    STYLECLIP_DIR = os.path.join(STYLEGAN_DIR, "StyleCLIP")
    GLOBAL_DIR = os.path.join(STYLECLIP_DIR, "global_directions")
    if GLOBAL_DIR not in sys.path: sys.path.append(GLOBAL_DIR)
    
    from models.psp import pSp
    fs3_path = os.path.join(GLOBAL_DIR, 'npy/ffhq/fs3.npy')
    try: fs3 = np.load(fs3_path)
    except: fs3 = None

    try:
        model_path = E4E_PATH 
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts['stylegan_weights'] = os.path.join(STYLEGAN_DIR, "checkpoint/stylegan2-ffhq-config-f.pt")
        from argparse import Namespace
        if isinstance(opts, dict): opts = Namespace(**opts)
        net = pSp(opts)
        net.eval().to(DEVICE)
    except Exception as e:
        print(f"\n❌ [ERROR] Failed to load e4e model: {e}")
        net = None
    return None, fs3, None, (256, 256), None, net

def make_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = torch.nan_to_num(tensor, 0.0)
    return (tensor.clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).permute(0, 2, 3, 1).numpy())

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(DEVICE)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE)
    return zeroshot_weights

def GetDt(classnames, model):
    portrait_templates = [
        "a photo of a {}.",
        "a high quality portrait of a {}.",
        "a realistic photograph of a {}.",
        "a close-up portrait of a {}."
    ]
    text_features = zeroshot_classifier(classnames, portrait_templates, model).t()
    dt = text_features[0] - text_features[1]
    dt = dt / torch.linalg.norm(dt)
    return dt

def GetBoundary(fs3, dt, threshold, device=DEVICE):
    dt = torch.nan_to_num(dt, 0.0)
    if isinstance(fs3, np.ndarray): fs3 = torch.from_numpy(fs3).to(device)
    fs3 = torch.nan_to_num(fs3, 0.0)
    dt = dt.view(-1)
    ds_imp = torch.matmul(fs3, dt.float())
    ds_imp = torch.where(torch.abs(ds_imp) < threshold, torch.zeros_like(ds_imp), ds_imp)
    boundary = []
    cursor = 0

    layer_channels = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 256, 11: 256, 12: 128, 13: 128, 14: 64, 15: 64, 16: 32, 17: 32}
    for i in range(18):
        target_dim = layer_channels.get(i, 512)
        chunk_src = ds_imp[cursor: cursor + 512]
        if chunk_src.numel() < 512: chunk_src = F.pad(chunk_src, (0, 512 - chunk_src.numel()))
        elif chunk_src.numel() > 512: chunk_src = chunk_src[:512]
        if target_dim == 512: chunk = chunk_src
        elif target_dim < 512: chunk = chunk_src[:target_dim]
        else: chunk = F.pad(chunk_src, (0, target_dim - 512))
        chunk = torch.nan_to_num(chunk, 0.0)
        chunk = chunk / (chunk.norm() + 1e-8)
        chunk = torch.clamp(chunk, -0.15, 0.15)
        boundary.append(chunk.to(device).detach())
        cursor += 512
    return boundary, None

class OpenAICLIPModel(nn.Module):
    def __init__(self, modelsize, num_labels=2):
        super(OpenAICLIPModel, self).__init__()
        if modelsize == "xxl": self.model, self.preprocess = clip.load("RN50x64", device="cuda")
        self.num_labels = num_labels
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs).float()
        out = self.relu(self.fc1(image_features))
        out = self.relu(self.fc2(image_features))
        out = self.fc3(out)
        return out

clip_transforms1 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
clip_transforms2 = transforms.Compose([
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

try:
    from model import Generator, ModulatedConv2d
    original_forward = ModulatedConv2d.forward
    def safe_forward(self, input, style, *args, **kwargs):
        style = torch.nan_to_num(style, 0.0) + 1e-8
        out = original_forward(self, input, style, *args, **kwargs)
        return torch.nan_to_num(out, 0.0)
    ModulatedConv2d.forward = safe_forward
    g_ema = Generator(1024, 512, 8)
    ckpt = torch.load("/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt", map_location=DEVICE)
    state_dict = ckpt["g_ema"] if "g_ema" in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    g_ema.load_state_dict(new_state_dict, strict=False)
    g_ema.eval().to(DEVICE)
except Exception as e:
    print(f"\n🚨 Failed to load g_ema (StyleGAN2 Generator): {e}")
    g_ema = None

try:
    classifier = OpenAICLIPModel("xxl").to(DEVICE).eval()
    classifier.load_state_dict(torch.load('/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth'))
    criterion = nn.CrossEntropyLoss()
except Exception as e:
    print(f"\n🚨 Failed to load classifier (OpenAICLIPModel): {e}")
    classifier = None

try: 
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
except Exception as e: 
    print(f"\n🚨 Failed to load model (OpenAI CLIP ViT-B/32): {e}")
    model = None

try:
    _, fs3, _, _, _, net = generate_wplus()
    if fs3 is not None: fs3 = torch.from_numpy(fs3).float().to(DEVICE)
except Exception as e:
    print(f"\n🚨 Failed to execute generate_wplus() and load net (e4e pSp): {e}")
    fs3, net = None, None

percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True).to(DEVICE)

def to_01(tensor_in):
    return (tensor_in.clamp(-1.0, 1.0) + 1.0) / 2.0

def get_radial_masks(ph, pw, device):
    yy, xx = torch.meshgrid(torch.arange(ph, device=device), torch.arange(pw, device=device), indexing="ij")
    cy, cx = (ph - 1) / 2, (pw - 1) / 2
    dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    dist = dist / dist.max()
    low_mask  = (dist <= 0.25).float()[None, None, :, :]
    mid_mask  = ((dist > 0.25) & (dist <= 0.65)).float()[None, None, :, :]
    high_mask = (dist > 0.65).float()[None, None, :, :]
    return low_mask, mid_mask, high_mask

def compute_masked_stats(patches, mask):
    eps = 1e-6
    X = patches * mask
    cnt = mask.sum() + eps
    s1 = X.sum(dim=[2, 3])
    mean = s1 / cnt
    xc = X - mean[:, :, None, None]
    var = (xc * xc).sum(dim=[2, 3]) / cnt
    # Ensure standard deviation doesn't divide by zero
    skew = ((xc**3).sum(dim=[2, 3]) / cnt) / (torch.sqrt(var)**3 + eps)
    kurt = ((xc**4).sum(dim=[2, 3]) / cnt) / (var**2 + eps)
    return torch.stack([mean, var, skew, kurt], dim=-1)

def extract_radial_dct_stats(tensor_in, patch_size=32):
    gray = 0.299 * tensor_in[:, 0:1] + 0.587 * tensor_in[:, 1:2] + 0.114 * tensor_in[:, 2:3]
    b, c, h, w = gray.shape
    hc, wc = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    patches = gray[:, :, :hc, :wc].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, -1, patch_size, patch_size)
    
    F_dct = torch.abs(dct.dct_2d(patches, norm='ortho'))
    low_mask, mid_mask, high_mask = get_radial_masks(patch_size, patch_size, tensor_in.device)
    
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.mean(dim=1).view(b, -1)

def extract_global_radial_dct_stats(tensor_in):
    """
    Computes DCT statistics (including 4th-order kurtosis) across the entire unpatched image
    to ensure the macroscopic frequency profile aligns with authentic data.
    """
    gray = 0.299 * tensor_in[:, 0:1] + 0.587 * tensor_in[:, 1:2] + 0.114 * tensor_in[:, 2:3]
    b, c, h, w = gray.shape
    
    # Compute DCT on the full image
    F_dct = torch.abs(dct.dct_2d(gray, norm='ortho'))
    
    # Get radial masks for the full resolution
    low_mask, mid_mask, high_mask = get_radial_masks(h, w, tensor_in.device)
    
    # Compute stats (F_dct shape is B, 1, H, W)
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.view(b, -1)

def get_detector_dct_embedding(img_tensor_1024):
    img_320 = F.interpolate(img_tensor_1024, size=(320, 320), mode='bicubic', align_corners=False)
    img_norm = clip_transforms2(img_320)
    
    gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
    gray = (gray * 2.0) - 1.0
    
    dct_feats = dct.dct_2d(gray, norm='ortho')
    dct_reshaped = dct_feats.reshape(dct_feats.size(0), -1)
    dct_log = torch.log(torch.abs(dct_reshaped) + 1e-12)
    
    mean = dct_log.mean(dim=1, keepdim=True)
    std = dct_log.std(dim=1, keepdim=True) + 1e-8
    return (dct_log - mean) / std

# ==================================================================================
# --- ARCHITECTURAL ALIGNMENT FOR SR RESIDUALS ---
# ==================================================================================
class SRSurrogate(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

def get_aligned_sr_residual(img, scale, sr_net):
    lr = F.avg_pool2d(img, kernel_size=scale, stride=scale)
    sr_reconstruction = sr_net(lr)
    if sr_reconstruction.shape != img.shape:
        sr_reconstruction = F.interpolate(sr_reconstruction, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
    return torch.abs(img - sr_reconstruction)


if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default="results")        
    parser.add_argument("--inputpath", type=str, required=True)
    parser.add_argument("--realpath", type=str, default="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real")

    parser.add_argument("--plosscoeff", type=float, default=0.5) 
    parser.add_argument("--alpha", type=float, default=2.5) 
    parser.add_argument("--beta", type=float, default=0.12) 
    parser.add_argument("--lr", type=float, default=0.005)
    
    # Hyperparameters tuned for hybrid evasion
    parser.add_argument("--epsilon", type=float, default=32.0/255.0)
    parser.add_argument("--adv_step", type=float, default=2.0/255.0)
    parser.add_argument("--ig_steps", type=int, default=7)
    parser.add_argument("--cow_beta", type=float, default=0.75)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument('--sr-scale', type=int, default=4)
    args = parser.parse_args() 

    print("[INFO] Loading Ensemble Surrogates (ResNet50 & EfficientNet)...")
    try:
        surrogate_resnet = models.resnet50(pretrained=True).to(DEVICE).eval()
        surrogate_efficientnet = models.efficientnet_b0(pretrained=True).to(DEVICE).eval()
        sr_net = SRSurrogate(scale_factor=args.sr_scale).to(DEVICE).eval()
    except Exception as e:
        print(f"🚨 Failed to load ensemble models: {e}")
        sys.exit(1)

    print("[INFO] Loading Architectural Surrogate (Grey/Black-Box Configuration)...")
    try:
        openclip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
        openclip_model = openclip_model.to(DEVICE).eval()
        openclip_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
        
        # Instantiate the detector architecture WITHOUT loading trained weights
        target_detector = DetectorNet(openclip_model, TransformerModel).to(DEVICE)
        target_detector.eval()
        print("✅ Architectural Surrogate (Random Weights) Loaded Successfully.")
    except Exception as e:
        print(f"🚨 Failed to load Architectural Surrogate: {e}")
        sys.exit(1)

    if None in [g_ema, classifier, model, net]:
        print("\n🚨 FATAL ERROR: Foundation models failed to load. See the specific exception printouts above.")
        sys.exit(1)

    print("[INFO] Loading VLM (BLIP) for stochastic semantic targeting...")
    try:
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
        vlm_model.eval()
        print("✅ VLM successfully loaded.")
    except Exception as e:
        print(f"🚨 Failed to load VLM: {e}")
        sys.exit(1)

    # =========================================================
    # PRE-COMPUTE REAL IMAGE CLIP FEATURES (GLOBAL + LOCAL)
    # =========================================================
    print(f"[INFO] Scanning Real Image database from: {args.realpath}")
    real_imagelist = [f for f in os.listdir(args.realpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    real_global_embeddings = []
    real_local_embeddings = []
    real_paths = []
    
    with torch.no_grad():
        for r_img in tqdm(real_imagelist[:1000], desc="Encoding Real Reference Images"):
            r_path = os.path.join(args.realpath, r_img)
            try:
                img_real = Image.open(r_path).convert("RGB")
                img_r_clip = clip_transforms1(img_real).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
                
                # Global Embedding
                img_r_clip_norm = clip_transforms2(img_r_clip)
                emb_global = model.encode_image(img_r_clip_norm)
                emb_global /= emb_global.norm(dim=-1, keepdim=True)
                real_global_embeddings.append(emb_global)
                
                # Local Embedding (Shuffled Patches)
                img_r_shuffled = shuffle_patches_tensor(img_r_clip, grid=4)
                img_r_shuffled_norm = clip_transforms2(img_r_shuffled)
                emb_local = model.encode_image(img_r_shuffled_norm)
                emb_local /= emb_local.norm(dim=-1, keepdim=True)
                real_local_embeddings.append(emb_local)
                
                real_paths.append(r_path)
            except Exception:
                continue
                
    real_embeddings_tensor = torch.cat(real_global_embeddings, dim=0) 
    print(f"[INFO] Successfully loaded {len(real_paths)} real reference images with Global & Local contexts.")

    INPUTIMGPATH = args.inputpath 
    DESTROOT_DIR = os.path.join(args.savepath, "Hybrid_MIGCOW_Attack") 
    DEST_DIR_EVADED = os.path.join(DESTROOT_DIR, "evaded")
    # ---------------------------------------------------------
    # NEW: ADDED DIRECTORY FOR PHASE 1
    # ---------------------------------------------------------
    DEST_DIR_PHASE1 = os.path.join(DESTROOT_DIR, "phase1_semantic_GAN_BLIP")  
    os.makedirs(DEST_DIR_PHASE1, exist_ok=True)  
    os.makedirs(DEST_DIR_EVADED, exist_ok=True)  

    imagelist = [f for f in os.listdir(INPUTIMGPATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    SEMANTIC_ITERATIONS = 50 
    MIGCOW_ITERATIONS = 100 
    
    transform_e4e = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_lpips = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Detector specific transform to mimic pre-processing perfectly
    detector_transform = transforms.Compose([
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    for xind, imgfile in enumerate(imagelist):
        print(f'\nProcessing: {imgfile}')
        try:
            original_image = Image.open(os.path.join(INPUTIMGPATH, imgfile)).convert("RGB")
            
            print("  [VLM] Generating stochastic captions for aggregated semantic direction...")
            dt_list = []
            
            # Generate 5 stochastic caption pairs
            for i in range(5):
                with torch.no_grad():
                    vlm_inputs = vlm_processor(images=original_image, return_tensors="pt").to(DEVICE)
                    # Stochastic generation for base description
                    out = vlm_model.generate(
                        **vlm_inputs, 
                        max_new_tokens=30, 
                        do_sample=True, 
                        top_p=0.9, 
                        temperature=1.0
                    )
                    base_desc = vlm_processor.decode(out[0], skip_special_tokens=True)
                    
                    # Stochastic generation for target description
                    conditional_prompt = f"A photograph of {base_desc}, featuring"
                    vlm_modifier_inputs = vlm_processor(images=original_image, text=conditional_prompt, return_tensors="pt").to(DEVICE)
                    out_mod = vlm_model.generate(
                        **vlm_modifier_inputs, 
                        max_new_tokens=20,
                        do_sample=True,
                        top_p=0.9,
                        temperature=1.0
                    )
                    target_desc = vlm_processor.decode(out_mod[0], skip_special_tokens=True)
                    
                print(f"    - Sample {i+1} | Base: '{base_desc}' -> Target: '{target_desc}'")
                
                # Calculate individual direction vector
                dt_i = GetDt([target_desc, base_desc], model).detach()
                dt_list.append(dt_i)
            
            # Aggregate semantic directions to combat stochastic fluctuation
            dt_aggregated = sum(dt_list) / len(dt_list)
            dt = dt_aggregated / torch.linalg.norm(dt_aggregated)
            
            # --- Extract the exact text representation the detector relies on ---
            filename_caption = os.path.splitext(imgfile)[0].replace('_', ' ')
            detector_text_tokens = openclip_tokenizer(list([filename_caption]), context_length=77).to(DEVICE)

            with torch.no_grad():
                img_ref_clip = clip_transforms1(original_image).unsqueeze(0).to(DEVICE)
                img_ref_clip = clip_transforms2(img_ref_clip)
                src_img_features = model.encode_image(img_ref_clip)
                src_img_features = src_img_features / src_img_features.norm(dim=-1, keepdim=True)

            # =========================================================
            # FIND CLOSEST REAL IMAGE MATCH & EXTRACT TARGETS
            # =========================================================
            with torch.no_grad():
                # Match based on Global Features
                sims = torch.cosine_similarity(src_img_features, real_embeddings_tensor)
                best_idx = sims.argmax().item()
                best_real_path = real_paths[best_idx]
                
                real_img_pil = Image.open(best_real_path).convert("RGB")
                real_tensor_01 = transforms.ToTensor()(real_img_pil).unsqueeze(0).to(DEVICE)
                
                real_tensor_1024 = F.interpolate(real_tensor_01, size=(1024, 1024), mode='bicubic', align_corners=False)
                real_tensor_320 = F.interpolate(real_tensor_01, size=(320, 320), mode='bicubic', align_corners=False)
                
                # Global DCT Log-embedding
                target_global_dct_embedding = get_detector_dct_embedding(real_tensor_1024).detach()
                
                # SR Residuals (Detector 2 killer)
                real_sr_residual = get_aligned_sr_residual(real_tensor_1024, args.sr_scale, sr_net).detach()
                target_stats_sr_radial = extract_radial_dct_stats(real_sr_residual).detach()

                # ---- CONSTRAINT 3: TARGET REAL KURTOSIS EXTRACTION (LOCAL + GLOBAL) ----
                target_real_stats_local = extract_radial_dct_stats(real_tensor_1024).detach()
                target_real_kurtosis_local = target_real_stats_local[:, [3, 7, 11]]
                
                target_real_stats_global = extract_global_radial_dct_stats(real_tensor_1024).detach()
                target_real_kurtosis_global = target_real_stats_global[:, [3, 7, 11]]

                # Authentic Global Visual Target (OpenCLIP 320x320)
                target_global_openclip_embed = openclip_model.encode_image(clip_transforms2(real_tensor_320))
                target_global_openclip_embed /= target_global_openclip_embed.norm(dim=-1, keepdim=True)
                target_global_openclip_embed = target_global_openclip_embed.detach()
                
                # Authentic Local Visual Target (Shuffled Patches 320x320)
                real_tensor_320_shuffled = shuffle_patches_tensor(real_tensor_320, grid=4)
                target_local_openclip_embed = openclip_model.encode_image(clip_transforms2(real_tensor_320_shuffled))
                target_local_openclip_embed /= target_local_openclip_embed.norm(dim=-1, keepdim=True)
                target_local_openclip_embed = target_local_openclip_embed.detach()

            input_image_e4e = transform_e4e(original_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, latents = net(input_image_e4e, randomize_noise=False, return_latents=True)
                latent_in, latent_init = latents.detach().clone(), latents.detach().clone() 

            latent_in.requires_grad = True 
            optimizer_latent = optim.Adam([latent_in], lr=args.lr)
            img_src_fake_256 = transform_lpips(original_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad(): noises = g_ema.make_noise()
            boundary_curr, _ = GetBoundary(fs3, dt, threshold=args.beta)
            current_alpha = args.alpha
            
            # =========================================================
            # PHASE 1: SEMANTIC OPTIMIZATION
            # =========================================================
            print("  -> Executing Semantic Shift...")
            for tempi in range(SEMANTIC_ITERATIONS): 
                img_gen_base_raw, _ = g_ema([latent_in], input_is_latent=True, noise=noises, boundary_tmp2=boundary_curr, alpha=current_alpha, use_dt=True)  
                if img_gen_base_raw.shape[0] == g_ema.n_latent and img_src_fake_256.shape[0] == 1: 
                    img_gen_base_raw = img_gen_base_raw[-1].unsqueeze(0)

                img_gen_256 = F.interpolate(img_gen_base_raw, size=(256, 256), mode='area')
                p_loss = percept(img_gen_256, img_src_fake_256).sum()
                id_loss = F.mse_loss(latent_in, latent_init)

                img_gen_clip_norm = clip_transforms2((F.interpolate(img_gen_base_raw, 224).clamp(-1, 1) + 1) / 2.0)
                gen_img_features = model.encode_image(img_gen_clip_norm)
                gen_img_features = gen_img_features / gen_img_features.norm(dim=-1, keepdim=True)
                loss_direction = 1 - torch.cosine_similarity(gen_img_features - src_img_features, dt.unsqueeze(0), dim=1).mean()
                
                loss_semantic = args.plosscoeff * p_loss + 10.0 * id_loss + 1.5 * loss_direction
                
                optimizer_latent.zero_grad()
                loss_semantic.backward()
                torch.nn.utils.clip_grad_norm_([latent_in], max_norm=1.0)
                optimizer_latent.step()
                
                with torch.no_grad(): 
                    latent_in.clamp_(-14, 14)

            # =========================================================
            # PHASE 2: UNIFIED OMNI-ATTACK (ARCHITECTURAL SURROGATE)
            # =========================================================
            print("  -> Executing Architectural Surrogate Fusion Attack...")
            with torch.no_grad():
                img_gen_base_raw, _ = g_ema([latent_in], input_is_latent=True, noise=noises, boundary_tmp2=boundary_curr, alpha=current_alpha, use_dt=True)
                if img_gen_base_raw.shape[0] == g_ema.n_latent:
                    img_gen_base_raw = img_gen_base_raw[-1].unsqueeze(0)

            base_detached = img_gen_base_raw.detach()
            # ---------------------------------------------------------
            # SAVE PHASE 1 INTERMEDIATE RESULT
            # ---------------------------------------------------------
            base_name = os.path.splitext(imgfile)[0]
            img_ar_phase1 = make_image(base_detached)
            img_name_phase1 = os.path.join(DEST_DIR_PHASE1, f"phase1_{base_name}.png")
            Image.fromarray(img_ar_phase1[0]).save(img_name_phase1, format="PNG")
            print(f"  -> Saved Phase 1 semantic shift result to {img_name_phase1}")


            delta = torch.zeros(1, 3, 1024, 1024, device=DEVICE)
            g_momentum = torch.zeros_like(delta)

            num_patches = 4
            patch_size = 224

            def tv_loss(d_in):
                tv_h = torch.sum(torch.abs(d_in[:, :, 1:, :] - d_in[:, :, :-1, :]))
                tv_w = torch.sum(torch.abs(d_in[:, :, :, 1:] - d_in[:, :, :, :-1]))
                return (tv_h + tv_w) / (d_in.shape[2] * d_in.shape[3])
             
            def scrambler_loss(d_in, block_size=8):
                b, c, h, w = d_in.shape
                blocks = d_in.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
                blocks = blocks.contiguous().view(b, c, -1, block_size * block_size)
                block_vars = torch.var(blocks, dim=-1)
                var_of_vars = torch.var(block_vars, dim=-1).mean()
                return -var_of_vars

            def loss_fn_cls(d_in):
                img_final = base_detached + d_in
                img_final = img_final + torch.randn_like(img_final) * 0.01
                total_loss = 0

                for _ in range(num_patches):
                    h = random.randint(0, 1024 - patch_size)
                    w = random.randint(0, 1024 - patch_size)
                    x_patch = img_final[:, :, h:h+patch_size, w:w+patch_size]
                    
                    x_cls_448 = F.interpolate(x_patch, size=(448, 448), mode='bilinear', align_corners=False)
                    x_cls_norm_448 = clip_transforms2(to_01(x_cls_448))

                    x_cls_224 = F.interpolate(x_patch, size=(224, 224), mode='bilinear', align_corners=False)
                    x_cls_norm_224 = clip_transforms2(to_01(x_cls_224))

                    logits_clip = classifier(x_cls_norm_448)
                    loss_clip = -logits_clip[:, 0].mean()   

                    logits_res = surrogate_resnet(x_cls_norm_224)
                    loss_res = -logits_res[:, 0].mean()

                    logits_eff = surrogate_efficientnet(x_cls_norm_224)
                    loss_eff = -logits_eff[:, 0].mean()

                    total_loss += (0.4 * loss_clip) + (0.3 * loss_res) + (0.3 * loss_eff)

                return (total_loss / num_patches) + (20.0 * tv_loss(d_in)) + (100.0 * scrambler_loss(d_in))

            # --- ARCHITECTURAL SURROGATE CROSS-ATTENTION BREAK ---
            def loss_fn_architectural_surrogate(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                
                # 1. Transform exactly as the detector does
                img_norm = detector_transform(img_final_1024)
                
                # 2. Replicate the exact grayscale DCT pipeline
                gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
                gray = (gray * 2.0) - 1.0
                dct_feats = dct.dct_2d(gray, norm='ortho')
                
                # 3. Forward pass through the exact Cross-Attention mechanism (using random weights)
                logits = target_detector(img_norm, detector_text_tokens, dct_feats)
                
                # 4. Force classification to 0 (Real)
                target_label = torch.tensor([0], device=DEVICE)
                return nn.CrossEntropyLoss()(logits, target_label)
            
            def loss_fn_sr_only(d_in):
                img_final = to_01(base_detached + d_in)
                residual = get_aligned_sr_residual(img_final, args.sr_scale, sr_net)
                return 50.0 * torch.mean(torch.abs(residual))

            # --- CONSTRAINT 3: TARGET 4TH ORDER KURTOSIS (LOCAL + GLOBAL) ---
            def loss_fn_kurtosis(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                
                adv_stats_local = extract_radial_dct_stats(img_final_1024)
                adv_kurtosis_local = adv_stats_local[:, [3, 7, 11]]
                loss_kurt_local = F.mse_loss(adv_kurtosis_local, target_real_kurtosis_local)
                
                adv_stats_global = extract_global_radial_dct_stats(img_final_1024)
                adv_kurtosis_global = adv_stats_global[:, [3, 7, 11]]
                loss_kurt_global = F.mse_loss(adv_kurtosis_global, target_real_kurtosis_global)
                
                return 100.0 * (0.5 * loss_kurt_local + 0.5 * loss_kurt_global)

            # --- CONSTRAINT 2: LOCAL VISUAL SEMANTICS VIA PATCH SHUFFLING ---
            def loss_fn_local_visual_semantics(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_320 = F.interpolate(img_final_1024, size=(320, 320), mode='bicubic', align_corners=False)
                
                # Apply identical grid shuffle to adversarial image
                img_shuffled_320 = shuffle_patches_tensor(img_final_320, grid=4)
                img_norm = clip_transforms2(img_shuffled_320)
                
                adv_embed = openclip_model.encode_image(img_norm)
                adv_embed = adv_embed / adv_embed.norm(dim=-1, keepdim=True)
                
                return 10.0 * (1.0 - torch.sum(adv_embed * target_local_openclip_embed, dim=-1).mean())

            # --- CONSTRAINT 4: PERCEPTUAL FIDELITY & SPATIAL SMOOTHNESS ---
            def loss_fn_perceptual_smoothness(d_in):
                # 1. Total Variation (Spatial Smoothness to kill ViT block grids)
                tv_h = torch.sum(torch.abs(d_in[:, :, 1:, :] - d_in[:, :, :-1, :]))
                tv_w = torch.sum(torch.abs(d_in[:, :, :, 1:] - d_in[:, :, :, :-1]))
                tv_penalty = (tv_h + tv_w) / (d_in.shape[2] * d_in.shape[3])
                
                # 2. LPIPS (Color and Texture preservation)
                img_final = to_01(base_detached + d_in)
                img_final_256 = F.interpolate(img_final, size=(256, 256), mode='area')
                base_256 = F.interpolate(to_01(base_detached), size=(256, 256), mode='area')
                p_loss = percept(img_final_256, base_256).sum()
                
                return (50.0 * p_loss) + (100.0 * tv_penalty)

            # ---------------------------------------------------------
            # 3. THE MIG-COW OPTIMIZATION LOOP
            # ---------------------------------------------------------
            base_name = os.path.splitext(imgfile)[0]

            for tempi in range(MIGCOW_ITERATIONS):
                delta_adv = delta.detach().requires_grad_(True)
                baseline_delta = torch.zeros_like(delta_adv)
                
                g_cls = integrated_gradients(delta_adv, baseline_delta, loss_fn_cls, steps=args.ig_steps)
                g_sr = integrated_gradients(delta_adv, baseline_delta, loss_fn_sr_only, steps=args.ig_steps)
                g_arch = integrated_gradients(delta_adv, baseline_delta, loss_fn_architectural_surrogate, steps=args.ig_steps)
                
                # New Gradients for Local Semantics & Kurtosis match
                g_vis_local = integrated_gradients(delta_adv, baseline_delta, loss_fn_local_visual_semantics, steps=args.ig_steps)
                g_kurt = integrated_gradients(delta_adv, baseline_delta, loss_fn_kurtosis, steps=args.ig_steps)
                g_percept = integrated_gradients(delta_adv, baseline_delta, loss_fn_perceptual_smoothness, steps=args.ig_steps)

                # Re-allocated Gradient Weights
                g_cow = compute_cow_gradient([
                    1.0 * g_cls,         
                    1.0 * g_sr,
                    5.0 * g_arch,       # Priority to disrupt the surrogate cross-attention architecture
                    2.0 * g_vis_local,  # High-priority on local patch-level authenticity
                    1.5 * g_kurt,       # Direct alignment to the real image's 4th-order statistical shape
                    3.0 * g_percept     # NEW: Force optimization to maintain LPIPS/TV fidelity
                ], beta=args.cow_beta)

                l1_norm = torch.norm(g_cow, p=1) + 1e-8
                g_momentum = args.mu * g_momentum + (g_cow / l1_norm)
                
                with torch.no_grad():
                    delta = delta - args.adv_step * torch.sign(g_momentum)
                    delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)

                if tempi % 20 == 0:  
                    with torch.no_grad():
                        img_current = to_01(base_detached + delta)
                        
                        img_norm = detector_transform(img_current)
                        gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
                        gray = (gray * 2.0) - 1.0
                        dct_feats = dct.dct_2d(gray, norm='ortho')
                        
                        logits = target_detector(img_norm, detector_text_tokens, dct_feats)
                        prob_fake = F.softmax(logits, dim=1)[0, 1].item() * 100.0
                        
                        current_tv = tv_loss(delta).item() * 100.0  
                        
                        # Monitor Kurtosis loss directly
                        current_stats_local = extract_radial_dct_stats(img_current)
                        current_stats_global = extract_global_radial_dct_stats(img_current)
                        
                        current_kurtosis_mse_local = F.mse_loss(current_stats_local[:, [3, 7, 11]], target_real_kurtosis_local).item() * 100.0
                        current_kurtosis_mse_global = F.mse_loss(current_stats_global[:, [3, 7, 11]], target_real_kurtosis_global).item() * 100.0
                        current_kurtosis_mse = (current_kurtosis_mse_local + current_kurtosis_mse_global) / 2.0
                        
                        print(f"  [Iter {tempi}] TV: {current_tv:.2f} | Kurt MSE: {current_kurtosis_mse:.4f} | Surrogate 'Fake' Confidence: {prob_fake:.2f}%")
                
                del g_cls, g_sr, g_arch, g_vis_local, g_kurt, g_percept, g_cow, delta_adv, baseline_delta

            img_gen_final = base_detached + delta
            img_ar = make_image(img_gen_final) 
            tempdir = os.path.join(DEST_DIR_EVADED, "final_results")
            os.makedirs(tempdir, exist_ok=True) 
            
            img_name = os.path.join(tempdir, f"evaded_{base_name}.png") 
            Image.fromarray(img_ar[0]).save(img_name, format="PNG") 

        except Exception as e:
            print(f"\n❌ Error processing image {imgfile}:")
            traceback.print_exc()
            continue
'''
# Fixed target CSV logic right before 
# the image processing loop to calculate the semantic direction (dt) 
# using the predefined strings.

import argparse
import math
import os
import sys
import types
import torch
from torch import optim
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models 
import lpips
import random
import numpy as np
import clip
import pandas as pd
import torch_dct as dct
import traceback
import open_clip

# Added BLIP & Diffusers imports
# from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import AutoencoderKL

# ==================================================================================
# ---- SURROGATE SRM HIGH-PASS FILTER BANK (FOR AIDE EVASION) ----
# ==================================================================================
class SurrogateHPF(nn.Module):
    """
    Simulates the Spatial Rich Model (SRM) filters used in the AIDE architecture.
    Uses fundamental spatial kernels to extract high-frequency residuals.
    """
    def __init__(self):
        super().__init__()
        # Laplacian (Omnidirectional edges/noise)
        laplacian = torch.tensor([[[[ 0.,  1.,  0.],
                                    [ 1., -4.,  1.],
                                    [ 0.,  1.,  0.]]]])
        # Sobel X & Y (Directional edges)
        sobel_x = torch.tensor([[[[-1.,  0.,  1.],
                                  [-2.,  0.,  2.],
                                  [-1.,  0.,  1.]]]])
        sobel_y = torch.tensor([[[[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]]]])
        
        # Expand to 3 channels (RGB)
        self.filters = torch.cat([laplacian, sobel_x, sobel_y], dim=0).repeat(1, 3, 1, 1)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.conv.weight = nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        return self.conv(x)


# ==================================================================================
# ---- PATCH SHUFFLING UTILITY FOR LOCAL FEATURE EXTRACTION ----
# ==================================================================================
def shuffle_patches_tensor(tensor_in, grid=4):
    """
    Divides the input image tensor into a grid x grid patches and shuffles them spatially.
    Forces the feature extractor to evaluate local texture/semantics without global structure.
    """
    B, C, H, W = tensor_in.shape
    ph, pw = H // grid, W // grid
    H_new, W_new = ph * grid, pw * grid
    
    # Crop to ensure perfect divisibility
    tensor_cropped = tensor_in[:, :, :H_new, :W_new]
    
    # Extract patches
    patches = tensor_cropped.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.contiguous().view(B, C, grid * grid, ph, pw)
    
    # Shuffle patches
    shuffled = torch.zeros_like(patches)
    for b in range(B):
        idx = torch.randperm(grid * grid, device=tensor_in.device)
        shuffled[b] = patches[b, :, idx, :, :]
        
    # Reassemble the image from shuffled patches
    shuffled = shuffled.view(B, C, grid, grid, ph, pw)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_new, W_new)
    
    if H_new != H or W_new != W:
        shuffled = F.pad(shuffled, (0, W - W_new, 0, H - H_new))
        
    return shuffled

# ==================================================================================
# ---- Transformer FUSION ATTACK (ARCHITECTURAL SURROGATE) ----
# ==================================================================================
class TransformerModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=32, dropout_rate=0.1, num_classes=2):
        super(TransformerModel, self).__init__()
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, y, z):
        x, attn_weights = self.transformer_block(x, y, z)
        x = x.permute(0, 2, 1)  
        x = self.global_avg_pool(x).squeeze(-1) 
        x = self.dropout(x)
        return self.classifier(x), attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim),
                                 nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, y, z):
        attn_output, attn_weights = self.attention(x, y, z)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output)), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim // num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)

    def forward(self, x, y, z, mask=None):
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 0, 2)
        attn_output, attn_weights = self.multihead_attn(x, y, z, attn_mask=mask)
        return attn_output.permute(1, 0, 2), attn_weights  

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.position_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.position_emb(positions)
        return x

def standard_scale(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

class DetectorNet(nn.Module):
    def __init__(self, CLIP_model, TransformerModelClass):
        super(DetectorNet, self).__init__()
        self.TransformerModel = TransformerModelClass(embed_dim=(768), num_heads=8)
        self.CLIP_model = CLIP_model
        self.DCT_Embedder = nn.Linear((320*320), 768, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, Images, Text_Encodings, DCT_features):
        img_embedding = self.CLIP_model.encode_image(Images)
        text_embedding = self.CLIP_model.encode_text(Text_Encodings)
        
        DCT_features_reshaped = DCT_features.reshape(DCT_features.size(0), -1)
        DCT_features_reshaped = torch.log(torch.abs(DCT_features_reshaped) + 1e-12)
        DCT_embedding = standard_scale(DCT_features_reshaped)
        
        DCT_embedding = self.relu(self.DCT_Embedder(DCT_embedding))
        
        combined_embedding = torch.stack([img_embedding, DCT_embedding, text_embedding], dim=1)        
        CrossAttention_out, _ = self.TransformerModel(combined_embedding, combined_embedding, combined_embedding)
        
        return CrossAttention_out

# ==================================================================================
# ---- 1. MIG-COW ALGORITHMS ----
# ==================================================================================
def integrated_gradients(x, baseline, loss_fn, steps=7):
    if torch.allclose(x, baseline):
        x_temp = x.detach().requires_grad_(True)
        loss = loss_fn(x_temp)
        return torch.autograd.grad(loss, x_temp)[0].detach()

    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]

    total_gradients = torch.zeros_like(x)
    for scaled_x in scaled_inputs:
        scaled_x = scaled_x.detach().requires_grad_(True)
        loss = loss_fn(scaled_x)
        grad = torch.autograd.grad(loss, scaled_x)[0]
        total_gradients += grad

    avg_grad = total_gradients / steps
    return avg_grad.detach()

def compute_cow_gradient(grads, beta=0.75, delta=1e-8):
    N = len(grads)
    
    sanitized_grads = [torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in grads]
    norm_grads = [g / (g.norm(p=2) + 1e-8) for g in sanitized_grads]
    
    g_common = torch.stack(norm_grads).mean(dim=0)
    
    model_grads = [g.flatten() for g in norm_grads]
    stacked_flat_grads = torch.stack(model_grads) 
    G_double = stacked_flat_grads.to(torch.float64)
    
    try:
        U, S, Vh = torch.linalg.svd(G_double, full_matrices=False)
        v_min = U[:, -1].to(torch.float32)
    except torch._C._LinAlgError:
        v_min = torch.ones(N, device=stacked_flat_grads.device) / math.sqrt(N)

    stacked_grads = torch.stack(norm_grads) 
    broadcast_shape = [-1] + [1] * (stacked_grads.dim() - 1)
    g_ens = (stacked_grads * v_min.view(*broadcast_shape)).sum(dim=0)

    proj_coef = (g_ens.flatten() @ g_common.flatten()) / (g_common.flatten().norm()**2 + 1e-8)
    proj = proj_coef * g_common
    g_orth = g_ens - proj

    g_combined = beta * g_common + (1 - beta) * g_orth
    return g_combined

# ==================================================================================
# ---- MAIN SETUP & SCRIPT ----
# ==================================================================================
PROJECT_ROOT = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack"
sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def make_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = torch.nan_to_num(tensor, 0.0)
    return (tensor.clamp_(min=0.0, max=1.0).mul(255).type(torch.uint8).permute(0, 2, 3, 1).numpy())

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(DEVICE)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE)
    return zeroshot_weights

def GetDt(classnames, model):
    portrait_templates = [
        "a photo of a {}.",
        "a high quality image of a {}.",
        "a realistic photograph of a {}.",
        "a close-up shot of a {}."
    ]
    text_features = zeroshot_classifier(classnames, portrait_templates, model).t()
    dt = text_features[0] - text_features[1]
    dt = dt / torch.linalg.norm(dt)
    return dt

class OpenAICLIPModel(nn.Module):
    def __init__(self, modelsize, num_labels=2):
        super(OpenAICLIPModel, self).__init__()
        if modelsize == "xxl": self.model, self.preprocess = clip.load("RN50x64", device="cuda")
        self.num_labels = num_labels
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs).float()
        out = self.relu(self.fc1(image_features))
        out = self.relu(self.fc2(image_features))
        out = self.fc3(out)
        return out

clip_transforms1 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
clip_transforms2 = transforms.Compose([
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

# Load General Domain Diffusion VAE
print("[INFO] Loading Stable Diffusion VAE for General Domain Inversion...")
try:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.eval()
    vae.requires_grad_(False)
except Exception as e:
    print(f"🚨 Failed to load SD VAE: {e}")
    sys.exit(1)

try:
    classifier = OpenAICLIPModel("xxl").to(DEVICE).eval()
    classifier.load_state_dict(torch.load('/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth'))
    criterion = nn.CrossEntropyLoss()
except Exception as e:
    print(f"\n🚨 Failed to load classifier (OpenAICLIPModel): {e}")
    classifier = None

try: 
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
except Exception as e: 
    print(f"\n🚨 Failed to load model (OpenAI CLIP ViT-B/32): {e}")
    model = None

percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True).to(DEVICE)
surrogate_hpf = SurrogateHPF().to(DEVICE).eval()

def to_01(tensor_in):
    return (tensor_in.clamp(-1.0, 1.0) + 1.0) / 2.0

def get_radial_masks(ph, pw, device):
    yy, xx = torch.meshgrid(torch.arange(ph, device=device), torch.arange(pw, device=device), indexing="ij")
    cy, cx = (ph - 1) / 2, (pw - 1) / 2
    dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    dist = dist / dist.max()
    low_mask  = (dist <= 0.25).float()[None, None, :, :]
    mid_mask  = ((dist > 0.25) & (dist <= 0.65)).float()[None, None, :, :]
    high_mask = (dist > 0.65).float()[None, None, :, :]
    return low_mask, mid_mask, high_mask

def compute_masked_stats(patches, mask=None):
    eps = 1e-6
    if mask is not None:
        X = patches * mask
        cnt = mask.sum() + eps
    else:
        X = patches
        cnt = patches.shape[-1] * patches.shape[-2] + eps
        
    s1 = X.sum(dim=[2, 3]) if mask is not None else X.sum(dim=-1)
    mean = s1 / cnt
    
    if mask is not None:
        xc = X - mean[:, :, None, None]
        var = (xc * xc).sum(dim=[2, 3]) / cnt
        skew = ((xc**3).sum(dim=[2, 3]) / cnt) / (torch.sqrt(var)**3 + eps)
        kurt = ((xc**4).sum(dim=[2, 3]) / cnt) / (var**2 + eps)
    else:
        xc = X - mean.unsqueeze(-1)
        var = (xc * xc).sum(dim=-1) / cnt
        skew = ((xc**3).sum(dim=-1) / cnt) / (torch.sqrt(var)**3 + eps)
        kurt = ((xc**4).sum(dim=-1) / cnt) / (var**2 + eps)
        
    return torch.stack([mean, var, skew, kurt], dim=-1)

# ---- DCT DOMAIN STATS ----
def extract_radial_dct_stats(tensor_in, patch_size=32):
    gray = 0.299 * tensor_in[:, 0:1] + 0.587 * tensor_in[:, 1:2] + 0.114 * tensor_in[:, 2:3]
    b, c, h, w = gray.shape
    hc, wc = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    patches = gray[:, :, :hc, :wc].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, -1, patch_size, patch_size)
    
    F_dct = torch.abs(dct.dct_2d(patches, norm='ortho'))
    low_mask, mid_mask, high_mask = get_radial_masks(patch_size, patch_size, tensor_in.device)
    
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.mean(dim=1).view(b, -1)

def extract_global_radial_dct_stats(tensor_in):
    gray = 0.299 * tensor_in[:, 0:1] + 0.587 * tensor_in[:, 1:2] + 0.114 * tensor_in[:, 2:3]
    b, c, h, w = gray.shape
    F_dct = torch.abs(dct.dct_2d(gray, norm='ortho'))
    low_mask, mid_mask, high_mask = get_radial_masks(h, w, tensor_in.device)
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.view(b, -1)

# ---- NEW: SPATIAL DOMAIN STATS (1st & 2nd ORDER ONLY) ----
def extract_spatial_stats_local(tensor_in, patch_size=32):
    """Calculates ONLY spatial mean and variance (1st & 2nd order) across local patches."""
    b, c, h, w = tensor_in.shape
    hc, wc = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    patches = tensor_in[:, :, :hc, :wc].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Flatten spatial dimensions of patches
    patches = patches.contiguous().view(b, c, -1, patch_size * patch_size)
    
    mean = patches.mean(dim=-1)
    var = patches.var(dim=-1, unbiased=False)
    
    stats = torch.stack([mean, var], dim=-1)
    stats_mean_across_patches = stats.mean(dim=2)
    return stats_mean_across_patches.view(b, -1)

def extract_spatial_stats_global(tensor_in):
    """Calculates ONLY global spatial mean and variance."""
    b, c, h, w = tensor_in.shape
    flat_image = tensor_in.view(b, c, -1)
    
    mean = flat_image.mean(dim=-1)
    var = flat_image.var(dim=-1, unbiased=False)
    
    stats = torch.stack([mean, var], dim=-1)
    return stats.view(b, -1)

def get_detector_dct_embedding(img_tensor_1024):
    img_320 = F.interpolate(img_tensor_1024, size=(320, 320), mode='bicubic', align_corners=False)
    img_norm = clip_transforms2(img_320)
    
    gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
    gray = (gray * 2.0) - 1.0
    
    dct_feats = dct.dct_2d(gray, norm='ortho')
    dct_reshaped = dct_feats.reshape(dct_feats.size(0), -1)
    dct_log = torch.log(torch.abs(dct_reshaped) + 1e-12)
    
    mean = dct_log.mean(dim=1, keepdim=True)
    std = dct_log.std(dim=1, keepdim=True) + 1e-8
    return (dct_log - mean) / std

# ==================================================================================
# --- CONTENT AWARE MASKING ---
# ==================================================================================
def get_content_mask(img_tensor, kernel_size=5):
    pad = kernel_size // 2
    mean = F.avg_pool2d(img_tensor, kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(img_tensor**2, kernel_size, stride=1, padding=pad)
    variance = torch.clamp(mean_sq - mean**2, min=0.0)
    mask = variance / (variance.amax(dim=(2, 3), keepdim=True) + 1e-8)
    mask = torch.sqrt(mask)
    return mask.detach()

class SRSurrogate(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

def get_aligned_sr_residual(img, scale, sr_net):
    lr = F.avg_pool2d(img, kernel_size=scale, stride=scale)
    sr_reconstruction = sr_net(lr)
    if sr_reconstruction.shape != img.shape:
        sr_reconstruction = F.interpolate(sr_reconstruction, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
    return torch.abs(img - sr_reconstruction)


if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default="results")        
    parser.add_argument("--inputpath", type=str, required=True)
    parser.add_argument("--realpath", type=str, default="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/SD_dataset/test/real")

    parser.add_argument("--plosscoeff", type=float, default=0.5) 
    parser.add_argument("--alpha", type=float, default=2.5) 
    parser.add_argument("--beta", type=float, default=0.12) 
    parser.add_argument("--lr", type=float, default=0.005)
    
    parser.add_argument("--epsilon", type=float, default=32.0/255.0)
    parser.add_argument("--adv_step", type=float, default=2.0/255.0)
    parser.add_argument("--ig_steps", type=int, default=7)
    parser.add_argument("--cow_beta", type=float, default=0.75)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument('--sr-scale', type=int, default=4)
    args = parser.parse_args() 

    print("[INFO] Loading Ensemble Surrogates (ResNet50 & EfficientNet)...")
    try:
        surrogate_resnet = models.resnet50(pretrained=True).to(DEVICE).eval()
        surrogate_efficientnet = models.efficientnet_b0(pretrained=True).to(DEVICE).eval()
        sr_net = SRSurrogate(scale_factor=args.sr_scale).to(DEVICE).eval()
    except Exception as e:
        print(f"🚨 Failed to load ensemble models: {e}")
        sys.exit(1)

    print("[INFO] Loading Architectural Surrogate (Grey/Black-Box Configuration)...")
    try:
        openclip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
        openclip_model = openclip_model.to(DEVICE).eval()
        openclip_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
        
        target_detector = DetectorNet(openclip_model, TransformerModel).to(DEVICE)
        target_detector.eval()
        print("✅ Architectural Surrogate (Random Weights) Loaded Successfully.")
    except Exception as e:
        print(f"🚨 Failed to load Architectural Surrogate: {e}")
        sys.exit(1)

    if None in [classifier, model, vae]:
        print("\n🚨 FATAL ERROR: Foundation models failed to load.")
        sys.exit(1)

    # ==================================================================================
    # --- COMMENTED OUT BLIP MODEL LOADING ---
    # print("[INFO] Loading VLM (BLIP) for stochastic semantic targeting...")
    # try:
    #     vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    #     vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
    #     vlm_model.eval()
    #     print("✅ VLM successfully loaded.")
    # except Exception as e:
    #     print(f"🚨 Failed to load VLM: {e}")
    #     sys.exit(1)
    # ==================================================================================

    print(f"[INFO] Scanning Real Image database from: {args.realpath}")
    real_imagelist = [f for f in os.listdir(args.realpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    real_global_embeddings = []
    real_paths = []
    
    with torch.no_grad():
        for r_img in tqdm(real_imagelist[:1000], desc="Encoding Real Reference Images"):
            r_path = os.path.join(args.realpath, r_img)
            try:
                img_real = Image.open(r_path).convert("RGB")
                img_r_clip = clip_transforms1(img_real).unsqueeze(0).to(DEVICE) 
                
                img_r_clip_norm = clip_transforms2(img_r_clip)
                emb_global = model.encode_image(img_r_clip_norm)
                emb_global /= emb_global.norm(dim=-1, keepdim=True)
                real_global_embeddings.append(emb_global)
                
                real_paths.append(r_path)
            except Exception:
                continue
                
    real_embeddings_tensor = torch.cat(real_global_embeddings, dim=0) 

    INPUTIMGPATH = args.inputpath 
    DESTROOT_DIR = os.path.join(args.savepath, "Hybrid_MIGCOW_Attack") 
    DEST_DIR_EVADED = os.path.join(DESTROOT_DIR, "evaded")  
    DEST_DIR_PHASE1 = os.path.join(DESTROOT_DIR, "phase1_semantic_GAN_FixedTarget")
    os.makedirs(DEST_DIR_EVADED, exist_ok=True)  
    os.makedirs(DEST_DIR_PHASE1, exist_ok=True)

    imagelist = [f for f in os.listdir(INPUTIMGPATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    SEMANTIC_ITERATIONS = 50 
    MIGCOW_ITERATIONS = 100 

    # --- NEW: Load fixed target descriptions ---
    print("[INFO] Loading fixed semantic targets...")
    try:
        dfstrings = pd.read_csv("./data/neutraltargets.csv")
        allneutrals, alltargets = dfstrings['neutral'].tolist(), dfstrings['target'].tolist()
    except:
        allneutrals, alltargets = ["neutral"], ["smiling"]
        print("  -> Could not load ./data/neutraltargets.csv. Defaulting to 'neutral' -> 'smiling'")
    
    transform_vae = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_lpips = transforms.Compose([
        transforms.Resize(512), 
        transforms.CenterCrop(512), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    detector_transform = transforms.Compose([
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    for xind, imgfile in enumerate(imagelist):
        print(f'\nProcessing: {imgfile}')
        try:
            original_image = Image.open(os.path.join(INPUTIMGPATH, imgfile)).convert("RGB")
            
            # ==================================================================================
            # --- COMMENTED OUT STOCHASTIC BLIP CAPTION GENERATION ---
            # print("  [VLM] Generating stochastic captions for aggregated semantic direction...")
            # dt_list = []
            # 
            # for i in range(5):
            #     with torch.no_grad():
            #         vlm_inputs = vlm_processor(images=original_image, return_tensors="pt").to(DEVICE)
            #         out = vlm_model.generate(**vlm_inputs, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=1.0)
            #         base_desc = vlm_processor.decode(out[0], skip_special_tokens=True)
            #         
            #         conditional_prompt = f"A photograph of {base_desc}, featuring"
            #         vlm_modifier_inputs = vlm_processor(images=original_image, text=conditional_prompt, return_tensors="pt").to(DEVICE)
            #         out_mod = vlm_model.generate(**vlm_modifier_inputs, max_new_tokens=20, do_sample=True, top_p=0.9, temperature=1.0)
            #         target_desc = vlm_processor.decode(out_mod[0], skip_special_tokens=True)
            # 
            #     print(f"    - Sample {i+1} | Base: '{base_desc}' -> Target: '{target_desc}'")
            #     dt_i = GetDt([target_desc, base_desc], model).detach()
            #     dt_list.append(dt_i)
            # ==================================================================================

            print("  [Fixed Text] Calculating semantic direction from fixed targets...")
            dt_list = []
            for base_desc, target_desc in zip(allneutrals, alltargets):
                dt_i = GetDt([target_desc, base_desc], model).detach()
                dt_list.append(dt_i)
            
            dt_aggregated = sum(dt_list) / len(dt_list)
            dt = dt_aggregated / torch.linalg.norm(dt_aggregated)
            
            filename_caption = os.path.splitext(imgfile)[0].replace('_', ' ')
            detector_text_tokens = openclip_tokenizer(list([filename_caption]), context_length=77).to(DEVICE)

            with torch.no_grad():
                img_ref_clip = clip_transforms1(original_image).unsqueeze(0).to(DEVICE)
                img_ref_clip = clip_transforms2(img_ref_clip)
                src_img_features = model.encode_image(img_ref_clip)
                src_img_features = src_img_features / src_img_features.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                sims = torch.cosine_similarity(src_img_features, real_embeddings_tensor)
                best_idx = sims.argmax().item()
                best_real_path = real_paths[best_idx]
                
                real_img_pil = Image.open(best_real_path).convert("RGB")
                real_tensor_01 = transforms.ToTensor()(real_img_pil).unsqueeze(0).to(DEVICE)
                
                real_tensor_1024 = F.interpolate(real_tensor_01, size=(1024, 1024), mode='bicubic', align_corners=False)
                real_tensor_320 = F.interpolate(real_tensor_01, size=(320, 320), mode='bicubic', align_corners=False)
                
                target_real_srm = surrogate_hpf(real_tensor_1024).detach()
                target_real_convnext = openclip_model.encode_image(clip_transforms2(real_tensor_320)).detach()
                target_real_convnext /= target_real_convnext.norm(dim=-1, keepdim=True)
                
                real_sr_residual = get_aligned_sr_residual(real_tensor_1024, args.sr_scale, sr_net).detach()
                
                target_real_stats_local = extract_radial_dct_stats(real_tensor_1024).detach()
                target_real_kurtosis_local = target_real_stats_local[:, [3, 7, 11]]
                
                target_real_stats_global = extract_global_radial_dct_stats(real_tensor_1024).detach()
                target_real_kurtosis_global = target_real_stats_global[:, [3, 7, 11]]

                # Target Stats: Spatial Domain
                target_real_spatial_local = extract_spatial_stats_local(real_tensor_1024).detach()
                target_real_spatial_global = extract_spatial_stats_global(real_tensor_1024).detach()

                target_local_openclip_embed = openclip_model.encode_image(clip_transforms2(shuffle_patches_tensor(real_tensor_320, grid=4)))
                target_local_openclip_embed /= target_local_openclip_embed.norm(dim=-1, keepdim=True)
                target_local_openclip_embed = target_local_openclip_embed.detach()

            input_image_vae = transform_vae(original_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                latent_init = vae.encode(input_image_vae).latent_dist.sample()
                latent_init = latent_init * vae.config.scaling_factor
                latent_in = latent_init.detach().clone()

            latent_in.requires_grad = True 
            optimizer_latent = optim.Adam([latent_in], lr=args.lr)
            img_src_fake_512 = transform_lpips(original_image).unsqueeze(0).to(DEVICE)
            
            print("  -> Executing Semantic Shift via VAE Latent Optimization...")
            for tempi in range(SEMANTIC_ITERATIONS): 
                img_gen_base_raw = vae.decode(latent_in / vae.config.scaling_factor).sample
                
                p_loss = percept(img_gen_base_raw, img_src_fake_512).sum()
                id_loss = F.mse_loss(latent_in, latent_init)

                img_gen_clip_norm = clip_transforms2((F.interpolate(img_gen_base_raw, 224).clamp(-1, 1) + 1) / 2.0)
                gen_img_features = model.encode_image(img_gen_clip_norm)
                gen_img_features = gen_img_features / gen_img_features.norm(dim=-1, keepdim=True)
                loss_direction = 1 - torch.cosine_similarity(gen_img_features - src_img_features, dt.unsqueeze(0), dim=1).mean()
                
                loss_semantic = args.plosscoeff * p_loss + 10.0 * id_loss + 1.5 * loss_direction
                
                optimizer_latent.zero_grad()
                loss_semantic.backward()
                torch.nn.utils.clip_grad_norm_([latent_in], max_norm=1.0)
                optimizer_latent.step()

            print("  -> Executing Architectural Surrogate Fusion Attack (AIDE-Targeted)...")
            with torch.no_grad():
                img_gen_base_raw = vae.decode(latent_in / vae.config.scaling_factor).sample
                img_gen_base_raw = F.interpolate(img_gen_base_raw, size=(1024, 1024), mode='bicubic', align_corners=False)

            base_detached = img_gen_base_raw.detach()
            
            base_name = os.path.splitext(imgfile)[0]
            img_phase1_ar = make_image(to_01(base_detached))
            phase1_img_name = os.path.join(DEST_DIR_PHASE1, f"phase1_{base_name}.png")
            Image.fromarray(img_phase1_ar[0]).save(phase1_img_name, format="PNG")

            delta = torch.zeros(1, 3, 1024, 1024, device=DEVICE)
            g_momentum = torch.zeros_like(delta)

            num_patches = 4
            patch_size = 224

            def tv_loss(d_in):
                tv_h = torch.sum(torch.abs(d_in[:, :, 1:, :] - d_in[:, :, :-1, :]))
                tv_w = torch.sum(torch.abs(d_in[:, :, :, 1:] - d_in[:, :, :, :-1]))
                return (tv_h + tv_w) / (d_in.shape[2] * d_in.shape[3])

            def loss_fn_cls(d_in):
                img_final = base_detached + d_in
                img_final = img_final + torch.randn_like(img_final) * 0.01
                total_loss = 0

                for _ in range(num_patches):
                    h = random.randint(0, 1024 - patch_size)
                    w = random.randint(0, 1024 - patch_size)
                    x_patch = img_final[:, :, h:h+patch_size, w:w+patch_size]
                    
                    x_cls_448 = F.interpolate(x_patch, size=(448, 448), mode='bilinear', align_corners=False)
                    x_cls_norm_448 = clip_transforms2(to_01(x_cls_448))

                    x_cls_224 = F.interpolate(x_patch, size=(224, 224), mode='bilinear', align_corners=False)
                    x_cls_norm_224 = clip_transforms2(to_01(x_cls_224))

                    logits_clip = classifier(x_cls_norm_448)
                    loss_clip = -logits_clip[:, 0].mean()   

                    logits_res = surrogate_resnet(x_cls_norm_224)
                    loss_res = -logits_res[:, 0].mean()

                    logits_eff = surrogate_efficientnet(x_cls_norm_224)
                    loss_eff = -logits_eff[:, 0].mean()

                    total_loss += (0.4 * loss_clip) + (0.3 * loss_res) + (0.3 * loss_eff)

                return (total_loss / num_patches) + (20.0 * tv_loss(d_in))
             
            def loss_fn_architectural_surrogate(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_norm = detector_transform(img_final_1024)
                gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
                gray = (gray * 2.0) - 1.0
                dct_feats = dct.dct_2d(gray, norm='ortho')
                
                logits = target_detector(img_norm, detector_text_tokens, dct_feats)
                target_label = torch.tensor([0], device=DEVICE)
                return nn.CrossEntropyLoss()(logits, target_label)

            def loss_fn_srm(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                adv_srm = surrogate_hpf(img_final_1024)
                
                adv_srm_var = torch.var(adv_srm.view(adv_srm.size(0), 3, -1), dim=2)
                target_srm_var = torch.var(target_real_srm.view(target_real_srm.size(0), 3, -1), dim=2)
                
                adv_srm_mean = torch.mean(adv_srm.view(adv_srm.size(0), 3, -1), dim=2)
                target_srm_mean = torch.mean(target_real_srm.view(target_real_srm.size(0), 3, -1), dim=2)
                
                loss_var = F.mse_loss(adv_srm_var, target_srm_var)
                loss_mean = F.mse_loss(adv_srm_mean, target_srm_mean)
                
                return 50.0 * (loss_var + loss_mean)

            def loss_fn_convnext(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_320 = F.interpolate(img_final_1024, size=(320, 320), mode='bicubic', align_corners=False)
                img_norm = clip_transforms2(img_final_320)
                
                adv_convnext = openclip_model.encode_image(img_norm)
                adv_convnext = adv_convnext / adv_convnext.norm(dim=-1, keepdim=True)
                
                return 20.0 * (1.0 - torch.sum(adv_convnext * target_real_convnext, dim=-1).mean())

            def loss_fn_sr_only(d_in):
                img_final = to_01(base_detached + d_in)
                residual = get_aligned_sr_residual(img_final, args.sr_scale, sr_net)
                return 50.0 * torch.mean(torch.abs(residual))

            def loss_fn_kurtosis(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                
                adv_stats_local = extract_radial_dct_stats(img_final_1024)
                adv_kurtosis_local = adv_stats_local[:, [3, 7, 11]]
                loss_kurt_local = F.mse_loss(adv_kurtosis_local, target_real_kurtosis_local)
                
                adv_stats_global = extract_global_radial_dct_stats(img_final_1024)
                adv_kurtosis_global = adv_stats_global[:, [3, 7, 11]]
                loss_kurt_global = F.mse_loss(adv_kurtosis_global, target_real_kurtosis_global)
                
                return 100.0 * (0.5 * loss_kurt_local + 0.5 * loss_kurt_global)

            def loss_fn_local_visual_semantics(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_320 = F.interpolate(img_final_1024, size=(320, 320), mode='bicubic', align_corners=False)
                img_shuffled_320 = shuffle_patches_tensor(img_final_320, grid=4)
                img_norm = clip_transforms2(img_shuffled_320)
                
                adv_embed = openclip_model.encode_image(img_norm)
                adv_embed = adv_embed / adv_embed.norm(dim=-1, keepdim=True)
                
                return 10.0 * (1.0 - torch.sum(adv_embed * target_local_openclip_embed, dim=-1).mean())

            def loss_fn_spatial_stats(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                
                adv_spatial_local = extract_spatial_stats_local(img_final_1024)
                loss_spatial_local = F.mse_loss(adv_spatial_local, target_real_spatial_local)
                
                adv_spatial_global = extract_spatial_stats_global(img_final_1024)
                loss_spatial_global = F.mse_loss(adv_spatial_global, target_real_spatial_global)
                
                return 100.0 * (0.5 * loss_spatial_local + 0.5 * loss_spatial_global)

            content_mask = get_content_mask(base_detached)

            for tempi in range(MIGCOW_ITERATIONS):
                delta_adv = delta.detach().requires_grad_(True)
                baseline_delta = torch.zeros_like(delta_adv)
                
                g_cls = integrated_gradients(delta_adv, baseline_delta, loss_fn_cls, steps=args.ig_steps)
                g_sr = integrated_gradients(delta_adv, baseline_delta, loss_fn_sr_only, steps=args.ig_steps)
                g_arch = integrated_gradients(delta_adv, baseline_delta, loss_fn_architectural_surrogate, steps=args.ig_steps)
                g_vis_local = integrated_gradients(delta_adv, baseline_delta, loss_fn_local_visual_semantics, steps=args.ig_steps)
                g_kurt = integrated_gradients(delta_adv, baseline_delta, loss_fn_kurtosis, steps=args.ig_steps)
                g_srm_spoof = integrated_gradients(delta_adv, baseline_delta, loss_fn_srm, steps=args.ig_steps)
                g_convnext = integrated_gradients(delta_adv, baseline_delta, loss_fn_convnext, steps=args.ig_steps)
                g_spatial_stats = integrated_gradients(delta_adv, baseline_delta, loss_fn_spatial_stats, steps=args.ig_steps)

                g_cow = compute_cow_gradient([
                    1.0 * g_cls,         
                    1.0 * g_sr,
                    4.0 * g_arch,       
                    1.5 * g_vis_local,  
                    1.5 * g_kurt,
                    4.0 * g_srm_spoof,  
                    3.0 * g_convnext,
                    2.0 * g_spatial_stats 
                ], beta=args.cow_beta)

                l1_norm = torch.norm(g_cow, p=1) + 1e-8
                g_momentum = args.mu * g_momentum + (g_cow / l1_norm)
                
                with torch.no_grad():
                    step_mask = 0.3 + 0.7 * content_mask
                    delta = delta - args.adv_step * torch.sign(g_momentum) * step_mask
                    delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)

                if tempi % 20 == 0:  
                    with torch.no_grad():
                        img_current = to_01(base_detached + delta)
                        
                        img_norm = detector_transform(img_current)
                        gray = 0.2989 * img_norm[:, 0:1] + 0.5870 * img_norm[:, 1:2] + 0.1140 * img_norm[:, 2:3]
                        gray = (gray * 2.0) - 1.0
                        dct_feats = dct.dct_2d(gray, norm='ortho')
                        
                        logits = target_detector(img_norm, detector_text_tokens, dct_feats)
                        prob_fake = F.softmax(logits, dim=1)[0, 1].item() * 100.0
                        
                        current_tv = tv_loss(delta).item() * 100.0  
                        
                        current_stats_local = extract_radial_dct_stats(img_current)
                        current_stats_global = extract_global_radial_dct_stats(img_current)
                        current_kurtosis_mse_local = F.mse_loss(current_stats_local[:, [3, 7, 11]], target_real_kurtosis_local).item() * 100.0
                        current_kurtosis_mse_global = F.mse_loss(current_stats_global[:, [3, 7, 11]], target_real_kurtosis_global).item() * 100.0
                        current_kurtosis_mse = (current_kurtosis_mse_local + current_kurtosis_mse_global) / 2.0
                        
                        print(f"  [Iter {tempi}] TV: {current_tv:.2f} | Kurt MSE: {current_kurtosis_mse:.4f} | Surrogate 'Fake' Confidence: {prob_fake:.2f}%")
                
                del g_cls, g_sr, g_arch, g_vis_local, g_kurt, g_srm_spoof, g_convnext, g_spatial_stats, g_cow, delta_adv, baseline_delta

            img_gen_final = base_detached + delta
            img_ar = make_image(to_01(img_gen_final)) 
            tempdir = os.path.join(DEST_DIR_EVADED, "final_results")
            os.makedirs(tempdir, exist_ok=True) 
            
            img_name = os.path.join(tempdir, f"evaded_{base_name}.png") 
            Image.fromarray(img_ar[0]).save(img_name, format="PNG") 

        except Exception as e:
            print(f"\n❌ Error processing image {imgfile}:")
            traceback.print_exc()
            continue