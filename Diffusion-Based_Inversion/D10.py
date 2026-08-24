import argparse
import math
import os
import sys
import types
import torch
from torch import optim
from torch.nn import functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
import lpips
import random
import numpy as np
import clip
import traceback
import torch_dct as dct

# Diffusers for Phase 1 VAE & BLIP
from diffusers import AutoencoderKL
from transformers import BlipProcessor, BlipForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================================================================================
# ---- D3 TRANSFORMER ATTENTION ARCHITECTURE (INTEGRATED) ----
# ==================================================================================
class TransformerAttention(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim, last_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output

# ==================================================================================
# ---- MIG-COW ALGORITHMS ----
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
# ---- DCT & STATISTICAL EXTRACTION UTILITIES ----
# ==================================================================================
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
    cnt = mask.sum() * patches.shape[1] + eps 
    
    s1 = X.sum(dim=[1, 2, 3], keepdim=True)
    mean = s1 / cnt
    xc = X - mean
    
    var = (xc * xc * mask).sum(dim=[1, 2, 3], keepdim=True) / cnt
    skew = ((xc**3 * mask).sum(dim=[1, 2, 3], keepdim=True) / cnt) / (torch.sqrt(var)**3 + eps)
    kurt = ((xc**4 * mask).sum(dim=[1, 2, 3], keepdim=True) / cnt) / (var**2 + eps)
    
    return torch.cat([mean, var, skew, kurt], dim=-1).squeeze(1).squeeze(1)

def extract_radial_dct_stats(tensor_in, patch_size=32):
    b, c, h, w = tensor_in.shape 
    hc, wc = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    
    patches = tensor_in[:, :, :hc, :wc].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, -1, patch_size, patch_size)
    
    F_dct = torch.abs(dct.dct_2d(patches, norm='ortho'))
    
    low_mask, mid_mask, high_mask = get_radial_masks(patch_size, patch_size, tensor_in.device)
    
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.view(b, -1)

def extract_global_radial_dct_stats(tensor_in):
    b, c, h, w = tensor_in.shape 
    
    F_dct = torch.abs(dct.dct_2d(tensor_in, norm='ortho'))
    
    low_mask, mid_mask, high_mask = get_radial_masks(h, w, tensor_in.device)
    
    low_s = compute_masked_stats(F_dct, low_mask)
    mid_s = compute_masked_stats(F_dct, mid_mask)
    high_s = compute_masked_stats(F_dct, high_mask)
    
    feat = torch.cat([low_s, mid_s, high_s], dim=-1)
    return feat.view(b, -1)

# ==================================================================================
# ---- BLACK-BOX SURROGATE UTILITIES ----
# ==================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def seed_everything(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def make_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = torch.nan_to_num(tensor, 0.0)
    return (tensor.clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).permute(0, 2, 3, 1).numpy())

clip_transforms_d3 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

clip_normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], 
    std=[0.26862954, 0.26130258, 0.27577711]
)

def to_01(tensor_in):
    return (tensor_in.clamp(-1.0, 1.0) + 1.0) / 2.0

def get_low_pass_color(img_tensor):
    return transforms.GaussianBlur(kernel_size=21, sigma=5.0)(img_tensor)

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
        "an authentic, non-AI generated close-up shot of a {}."
    ]
    text_features = zeroshot_classifier(classnames, portrait_templates, model).t()
    dt = text_features[0] - text_features[1]
    dt = dt / torch.linalg.norm(dt)
    return dt

def d3_shuffle_patches_synchronized(x, patch_size=14, idx=None):
    B, C, H, W = x.size()
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
    if idx is None:
        idx = torch.randperm(patches.size(-1), device=x.device)
    shuffled_patches = patches[:, :, idx]
    shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return shuffled_images, idx

def shuffle_patches_tensor(tensor_in, grid=4):
    B, C, H, W = tensor_in.shape
    ph, pw = H // grid, W // grid
    H_new, W_new = ph * grid, pw * grid
    
    tensor_cropped = tensor_in[:, :, :H_new, :W_new]
    patches = tensor_cropped.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.contiguous().view(B, C, grid * grid, ph, pw)
    
    shuffled = torch.zeros_like(patches)
    for b in range(B):
        idx = torch.randperm(grid * grid, device=tensor_in.device)
        shuffled[b] = patches[b, :, idx, :, :]
        
    shuffled = shuffled.view(B, C, grid, grid, ph, pw)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_new, W_new)
    
    if H_new != H or W_new != W:
        shuffled = F.pad(shuffled, (0, W - W_new, 0, H - H_new))
        
    return shuffled

class PenultimateViTSurrogate(nn.Module):
    """
    Extracts the unpooled patch tokens from the penultimate layer of any ViT.
    This provides perfectly stable, differentiable gradients by avoiding 
    randomly initialized classification heads.
    """
    def __init__(self, model_name="ViT-L/14", device="cuda"):
        super(PenultimateViTSurrogate, self).__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.features = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            self.features = output.clone() 
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)

    def forward(self, x):
        self.model.encode_image(x.type(self.model.dtype))
        return self.features

# ==================================================================================
# ---- MAIN SCRIPT ----
# ==================================================================================
if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default="results")        
    parser.add_argument("--inputpath", type=str, required=True)
    parser.add_argument("--realpath", type=str, default="/forcolab/home/ashahj/D3/data/SD_dataset/test/real")

    # HYPERPARAMETERS 
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epsilon", type=float, default=24.0/255.0) 
    parser.add_argument("--adv_step", type=float, default=2.0/255.0)
    parser.add_argument("--ig_steps", type=int, default=7)
    parser.add_argument("--cow_beta", type=float, default=0.75)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--plosscoeff", type=float, default=0.5) 
    args = parser.parse_args() 

    print("[INFO] Loading Black-Box Feature Extractor (D3 Surrogate)...")
    try:
        # Load the stable, parameter-free feature extractor
        d3_surrogate = PenultimateViTSurrogate("ViT-L/14", DEVICE).to(DEVICE)
        print("✅ D3 Surrogate Loaded Successfully.")
    except Exception as e:
        print(f"🚨 Failed to load ViT Surrogate: {e}")
        sys.exit(1)

    print("[INFO] Loading Stable Diffusion VAE & BLIP VLM...")
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
        vae.requires_grad_(False)
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE).eval()
    except Exception as e:
        print(f"🚨 Failed to load VAE/VLM: {e}")
        sys.exit(1)

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True).to(DEVICE)

    # =========================================================
    # PRE-COMPUTE REAL IMAGE FEATURES
    # =========================================================
    print(f"[INFO] Scanning Real Image database from: {args.realpath}")
    real_imagelist = [f for f in os.listdir(args.realpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    real_global_embeddings = []
    real_paths = []
    
    with torch.no_grad():
        for r_img in tqdm(real_imagelist[:1000], desc="Encoding Real Reference Images"):
            r_path = os.path.join(args.realpath, r_img)
            try:
                img_real = Image.open(r_path).convert("RGB")
                img_r_norm = clip_transforms_d3(img_real).unsqueeze(0).to(DEVICE)
                
                # Extract 768-D projected features for Phase 1 Semantic Alignment
                src_img_features_768 = d3_surrogate.model.encode_image(img_r_norm.type(d3_surrogate.model.dtype))
                src_img_features_768 /= src_img_features_768.norm(dim=-1, keepdim=True)
                
                real_global_embeddings.append(src_img_features_768.float())
                real_paths.append(r_path)
            except Exception as e:
                continue
                
    real_embeddings_tensor = torch.cat(real_global_embeddings, dim=0) 

    INPUTIMGPATH = args.inputpath 
    DESTROOT_DIR = os.path.join(args.savepath, "BlackBox_D3_Discrepancy_Attack") 
    DEST_DIR_EVADED = os.path.join(DESTROOT_DIR, "evaded")  
    os.makedirs(DEST_DIR_EVADED, exist_ok=True)  

    imagelist = [f for f in os.listdir(INPUTIMGPATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    SEMANTIC_ITERATIONS = 40 
    MIGCOW_ITERATIONS = 120 
    
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

    # Global synchronization index for D3 patches
    MASTER_IDX = None

    for xind, imgfile in enumerate(imagelist):
        print(f'\nProcessing: {imgfile}')
        try:
            original_image = Image.open(os.path.join(INPUTIMGPATH, imgfile)).convert("RGB")
            
            # --- BLIP DYNAMIC CAPTIONING ---
            print("  [VLM] Generating stochastic captions for aggregated semantic shift...")
            dt_list = []
            for i in range(5):
                with torch.no_grad():
                    vlm_inputs = vlm_processor(images=original_image, return_tensors="pt").to(DEVICE)
                    out = vlm_model.generate(**vlm_inputs, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=1.0)
                    base_desc = vlm_processor.decode(out[0], skip_special_tokens=True)
                    
                    conditional_prompt = f"An authentic, non-AI generated photograph of {base_desc}, featuring"
                    vlm_modifier_inputs = vlm_processor(images=original_image, text=conditional_prompt, return_tensors="pt").to(DEVICE)
                    out_mod = vlm_model.generate(**vlm_modifier_inputs, max_new_tokens=20, do_sample=True, top_p=0.9, temperature=1.0)
                    target_desc = vlm_processor.decode(out_mod[0], skip_special_tokens=True)

                dt_i = GetDt([target_desc, base_desc], d3_surrogate.model).detach()
                dt_list.append(dt_i.float())
            
            dt_aggregated = sum(dt_list) / len(dt_list)
            dt = dt_aggregated / torch.linalg.norm(dt_aggregated)

            with torch.no_grad():
                img_ref_norm = clip_transforms_d3(original_image).unsqueeze(0).to(DEVICE)
                src_img_features_768 = d3_surrogate.model.encode_image(img_ref_norm.type(d3_surrogate.model.dtype))
                src_img_features_768 = src_img_features_768 / src_img_features_768.norm(dim=-1, keepdim=True)
                src_img_features_768 = src_img_features_768.float() 

            # =========================================================
            # EXTRACT TARGET REAL IMAGE DISCREPANCY & STATS
            # =========================================================
            with torch.no_grad():
                sims = torch.cosine_similarity(src_img_features_768, real_embeddings_tensor)
                best_idx = sims.argmax().item()
                best_real_path = real_paths[best_idx]
                
                real_img_pil = Image.open(best_real_path).convert("RGB")
                target_real_img_norm = clip_transforms_d3(real_img_pil).unsqueeze(0).to(DEVICE)
                
                # 1. Base D3 Target (Discrepancy)
                target_real_orig = d3_surrogate(target_real_img_norm).detach().float()

                # 2. Local Visual Semantics Target (Shuffled Grid)
                real_tensor_224 = F.interpolate(target_real_img_norm, size=(224, 224), mode='bicubic', align_corners=False)
                real_tensor_224_shuffled = shuffle_patches_tensor(real_tensor_224, grid=4)
                target_local_openclip_embed = d3_surrogate(real_tensor_224_shuffled).detach().float()
                target_local_openclip_embed = target_local_openclip_embed / target_local_openclip_embed.norm(dim=-1, keepdim=True)

                # 3. Kurtosis & Stat Targets
                target_real_stats_local = extract_radial_dct_stats(target_real_img_norm).detach()
                target_real_kurtosis_local = target_real_stats_local[:, [3, 7, 11]]
                
                target_real_stats_global = extract_global_radial_dct_stats(target_real_img_norm).detach()
                target_real_kurtosis_global = target_real_stats_global[:, [3, 7, 11]]

            # =========================================================
            # PHASE 1: VAE SEMANTIC SHIFT (MIG-COW Primer)
            # =========================================================
            input_image_vae = transform_vae(original_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                latent_init = vae.encode(input_image_vae).latent_dist.sample()
                latent_init = latent_init * vae.config.scaling_factor
                latent_in = latent_init.detach().clone()

            latent_in.requires_grad = True 
            optimizer_latent = optim.Adam([latent_in], lr=args.lr)
            img_src_fake_512 = transform_lpips(original_image).unsqueeze(0).to(DEVICE)
            
            print("  -> Phase 1: Semantic Shift (Pushing image toward Real Manifold)...")
            for tempi in range(SEMANTIC_ITERATIONS): 
                img_gen_base_raw = vae.decode(latent_in / vae.config.scaling_factor).sample
                p_loss = percept(img_gen_base_raw, img_src_fake_512).sum()
                id_loss = F.mse_loss(latent_in, latent_init)
                
                img_gen_224 = (F.interpolate(img_gen_base_raw, size=(224, 224), mode='bicubic', align_corners=False).clamp(-1, 1) + 1) / 2.0
                img_gen_clip_norm = clip_normalize(img_gen_224)
                
                gen_img_features_768 = d3_surrogate.model.encode_image(img_gen_clip_norm.type(d3_surrogate.model.dtype))
                gen_img_features_768 = gen_img_features_768 / gen_img_features_768.norm(dim=-1, keepdim=True)
                gen_img_features_768 = gen_img_features_768.float()
                
                loss_direction = 1 - torch.cosine_similarity(gen_img_features_768 - src_img_features_768, dt.unsqueeze(0), dim=1).mean()
                
                loss_semantic = args.plosscoeff * p_loss + 10.0 * id_loss + 2.0 * loss_direction
                
                optimizer_latent.zero_grad()
                loss_semantic.backward()
                torch.nn.utils.clip_grad_norm_([latent_in], max_norm=1.0)
                optimizer_latent.step()

            # =========================================================
            # PHASE 2: BLACK-BOX DISCREPANCY DISTILLATION + LOCAL/KURTOSIS
            # =========================================================
            print("  -> Phase 2: Executing Pure Black-Box Discrepancy Attack...")
            with torch.no_grad():
                img_gen_base_raw = vae.decode(latent_in / vae.config.scaling_factor).sample
                img_gen_base_raw = F.interpolate(img_gen_base_raw, size=(1024, 1024), mode='bicubic', align_corners=False)

            base_detached = img_gen_base_raw.detach()
            baseline_color = get_low_pass_color(to_01(base_detached)).detach()

            delta = torch.zeros(1, 3, 1024, 1024, device=DEVICE)
            g_momentum = torch.zeros_like(delta)

            def loss_fn_color(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                current_color = get_low_pass_color(img_final_1024)
                return 100.0 * F.l1_loss(current_color, baseline_color)

            def loss_fn_smoothness(d_in):
                tv_h = torch.sum(torch.abs(d_in[:, :, 1:, :] - d_in[:, :, :-1, :]))
                tv_w = torch.sum(torch.abs(d_in[:, :, :, 1:] - d_in[:, :, :, :-1]))
                return 50.0 * ((tv_h + tv_w) / (d_in.shape[2] * d_in.shape[3]))

            # --- LOCAL VISUAL SEMANTICS ---
            def loss_fn_local_visual_semantics(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_224 = F.interpolate(img_final_1024, size=(224, 224), mode='bicubic', align_corners=False)
                img_shuffled_224 = shuffle_patches_tensor(img_final_224, grid=4)
                img_norm = clip_normalize(img_shuffled_224)
                
                adv_embed = d3_surrogate(img_norm).float()
                adv_embed = adv_embed / adv_embed.norm(dim=-1, keepdim=True)
                return 10.0 * (1.0 - torch.sum(adv_embed * target_local_openclip_embed, dim=-1).mean())

            # --- 4TH-ORDER KURTOSIS ---
            def loss_fn_kurtosis(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_224 = F.interpolate(img_final_1024, size=(224, 224), mode='bicubic', align_corners=False)
                img_norm = clip_normalize(img_final_224)
                
                adv_stats_local = extract_radial_dct_stats(img_norm)
                adv_kurtosis_local = adv_stats_local[:, [3, 7, 11]]
                loss_kurt_local = F.mse_loss(adv_kurtosis_local, target_real_kurtosis_local)
                
                adv_stats_global = extract_global_radial_dct_stats(img_norm)
                adv_kurtosis_global = adv_stats_global[:, [3, 7, 11]]
                loss_kurt_global = F.mse_loss(adv_kurtosis_global, target_real_kurtosis_global)
                
                return 100.0 * (0.5 * loss_kurt_local + 0.5 * loss_kurt_global)

            # --- THE BLACK-BOX D3 DISCREPANCY SIMULATOR ---
            def loss_fn_d3_blackbox(d_in):
                img_final_1024 = to_01(base_detached + d_in)
                img_final_224 = F.interpolate(img_final_1024, size=(224, 224), mode='bicubic', align_corners=False)
                img_norm = clip_normalize(img_final_224)
                
                adv_orig = d3_surrogate(img_norm).float()
                
                real_shuffled, _ = d3_shuffle_patches_synchronized(target_real_img_norm, patch_size=14, idx=MASTER_IDX)
                target_real_shuf = d3_surrogate(real_shuffled).detach().float()
                
                img_shuffled, _ = d3_shuffle_patches_synchronized(img_norm, patch_size=14, idx=MASTER_IDX)
                adv_shuf = d3_surrogate(img_shuffled).float()
                
                adv_diff = adv_orig - adv_shuf
                real_diff = target_real_orig - target_real_shuf
                
                loss_diff_cos = 1.0 - F.cosine_similarity(adv_diff.flatten(1), real_diff.flatten(1), dim=-1).mean()
                loss_orig_cos = 1.0 - F.cosine_similarity(adv_orig.flatten(1), target_real_orig.flatten(1), dim=-1).mean()
                loss_shuf_cos = 1.0 - F.cosine_similarity(adv_shuf.flatten(1), target_real_shuf.flatten(1), dim=-1).mean()
                loss_orig_mse = F.mse_loss(adv_orig, target_real_orig)
                loss_shuf_mse = F.mse_loss(adv_shuf, target_real_shuf)

                return 200.0 * loss_diff_cos + 100.0 * (loss_orig_cos + loss_shuf_cos) + 50.0 * (loss_orig_mse + loss_shuf_mse)

            base_name = os.path.splitext(imgfile)[0]

            for tempi in range(MIGCOW_ITERATIONS):
                delta_adv = delta.detach().requires_grad_(True)
                baseline_delta = torch.zeros_like(delta_adv)
                
                # SYNCHRONIZE SHUFFLE INDEX FOR THIS ENTIRE GRADIENT STEP
                num_patches = (224 // 14) ** 2
                MASTER_IDX = torch.randperm(num_patches, device=DEVICE)

                # Compute integrated gradients
                g_col = integrated_gradients(delta_adv, baseline_delta, loss_fn_color, steps=args.ig_steps)
                g_tv = integrated_gradients(delta_adv, baseline_delta, loss_fn_smoothness, steps=args.ig_steps)
                g_d3_dist = integrated_gradients(delta_adv, baseline_delta, loss_fn_d3_blackbox, steps=args.ig_steps)
                g_vis_local = integrated_gradients(delta_adv, baseline_delta, loss_fn_local_visual_semantics, steps=args.ig_steps)
                g_kurt = integrated_gradients(delta_adv, baseline_delta, loss_fn_kurtosis, steps=args.ig_steps)

                # Assemble gradient trajectory 
                g_cow = compute_cow_gradient([
                    2.0 * g_col,     
                    0.1 * g_tv,       
                    25.0 * g_d3_dist, 
                    2.0 * g_vis_local,
                    1.5 * g_kurt
                ], beta=args.cow_beta)

                l1_norm = torch.norm(g_cow, p=1) + 1e-8
                g_momentum = args.mu * g_momentum + (g_cow / l1_norm)
                
                with torch.no_grad():
                    delta = delta - args.adv_step * torch.sign(g_momentum)
                    delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)

                if tempi % 20 == 0:  
                    with torch.no_grad():
                        dist_loss = loss_fn_d3_blackbox(delta).item() / 400.0
                        print(f"  [Iter {tempi}] D3 Black-Box Discrepancy Gap: {dist_loss:.4f}")

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