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
import lpips 
import random 
import numpy as np 
import clip 
import pandas as pd
import torch_dct as dct
import traceback
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==================================================================================
# ---- 1. MIG-COW ALGORITHMS ----
# ==================================================================================
def integrated_gradients(x, baseline, loss_fn, steps=7):
    # BUG FIX: If x and baseline are identical (iteration 0), attribution map is 0. 
    # We must return the raw gradient to jumpstart the optimizer.
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
    # BUG FIX: Return the average gradient for optimization, NOT the attribution map!
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
sys.modules["op"] = op_module
os.environ['STYLEGAN2_NO_CUSTOM_OPS'] = '1'

# ==================================================================================
# ---- MAIN SETUP & SCRIPT ----
# ==================================================================================
PROJECT_ROOT = "/path/to/your/EvolvingThreat-DeepfakeImageDetect/adversarialattack"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "encoder4editing"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "stylegan2-pytorch"))

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
        model_path = os.path.join(PROJECT_ROOT, "encoder4editing", "e4e_ffhq_encode.pt")
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts'] 
        opts['checkpoint_path'] = model_path
        opts['stylegan_weights'] = os.path.join(STYLEGAN_DIR, "checkpoint/stylegan2-ffhq-config-f.pt")
        from argparse import Namespace
        if isinstance(opts, dict): opts = Namespace(**opts)
        net = pSp(opts)
        net.eval().to(DEVICE)
    except Exception as e:
        print("\n❌ [ERROR] Failed to load e4e model:")
        net = None
    return None, fs3, None, (256,256), None, net

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
    portrait_templates = ["a photo of a {}.", "a high quality portrait of a {}.", "a realistic photograph of a {}.", "a close-up portrait of a {}."]
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
        if modelsize=="xxl": self.model, self.preprocess = clip.load("RN50x64", device="cuda")
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

clip_transforms1 = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
clip_transforms2 = transforms.Compose([transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])]) 

try:
    from model import Generator, ModulatedConv2d
    original_forward = ModulatedConv2d.forward
    def safe_forward(self, input, style, *args, **kwargs):
        style = torch.nan_to_num(style, 0.0) + 1e-8
        out = original_forward(self, input, style, *args, **kwargs)
        return torch.nan_to_num(out, 0.0)
    ModulatedConv2d.forward = safe_forward
    g_ema = Generator(1024, 512, 8)
    ckpt = torch.load("/path/to/your/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt", map_location=DEVICE)
    state_dict = ckpt["g_ema"] if "g_ema" in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    g_ema.load_state_dict(new_state_dict, strict=False)
    g_ema.eval().to(DEVICE)
except Exception as e:
    g_ema = None

try: 
    classifier = OpenAICLIPModel("xxl").to(DEVICE).eval()
    classifier.load_state_dict(torch.load('/path/to/your/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth'))
    criterion = nn.CrossEntropyLoss()
except Exception as e:
    classifier = None

try: model, preprocess = clip.load("ViT-B/32", device=DEVICE)
except Exception as e: model = None

try:
    _, fs3, _, _, _, net = generate_wplus()
    if fs3 is not None: fs3 = torch.from_numpy(fs3).float().to(DEVICE)
except Exception as e:
    fs3, net = None, None

percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True).to(DEVICE)

# ==================================================================================
# --- STATISTIC EXTRACTION (EXACT RADIAL BANDS) ---
# ==================================================================================
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

if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default="results")        
    parser.add_argument("--inputpath", type=str, required=True)
    parser.add_argument("--realpath", type=str, default="/path/to/your/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real")

    parser.add_argument("--plosscoeff", type=float, default=0.5) 
    parser.add_argument("--alpha", type=float, default=2.5) 
    parser.add_argument("--beta", type=float, default=0.12) 
    parser.add_argument("--lr", type=float, default=0.005)
    
    parser.add_argument("--epsilon", type=float, default=32.0/255.0)
    parser.add_argument("--adv_step", type=float, default=2.0/255.0)
    parser.add_argument("--ig_steps", type=int, default=7)
    parser.add_argument("--cow_beta", type=float, default=0.75)
    parser.add_argument("--mu", type=float, default=1.0)
    args = parser.parse_args() 

    if None in [g_ema, classifier, model, net]:
        print("\n🚨 FATAL ERROR: Foundation models failed to load.")
        sys.exit(1)

    print("[INFO] Loading VLM (BLIP) for dynamic semantic targeting...")
    vlm_model = None
    vlm_processor = None
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
        vlm_model.eval()
        print("[INFO] VLM successfully loaded.")
    except ImportError:
        print("\n⚠️ [WARNING] BLIP not found. Your 'transformers' library is outdated.")
    except Exception as e:
        print(f"\n⚠️ [WARNING] Failed to load VLM: {e}")

    try:
        dfstrings = pd.read_csv("./data/neutraltargets.csv")
        allneutrals, alltargets = dfstrings['neutral'].tolist(), dfstrings['target'].tolist()
    except:
        allneutrals, alltargets = ["neutral"], ["smiling"]

    # =========================================================
    # PRE-COMPUTE REAL IMAGE CLIP FEATURES
    # =========================================================
    print(f"[INFO] Scanning Real Image database from: {args.realpath}")
    real_imagelist = [f for f in os.listdir(args.realpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    real_embeddings, real_paths = [], []
    
    with torch.no_grad():
        for r_img in tqdm(real_imagelist[:999], desc="Encoding Real Reference Images"): # Limit to 500 for speed
            r_path = os.path.join(args.realpath, r_img)
            try:
                img_real = Image.open(r_path).convert("RGB")
                img_r_clip = clip_transforms1(img_real).unsqueeze(0).to(DEVICE)
                img_r_clip = clip_transforms2(img_r_clip)
                emb = model.encode_image(img_r_clip)
                emb /= emb.norm(dim=-1, keepdim=True)
                real_embeddings.append(emb)
                real_paths.append(r_path)
            except Exception:
                continue
                
    real_embeddings_tensor = torch.cat(real_embeddings, dim=0) 
    print(f"[INFO] Successfully loaded {len(real_paths)} real reference images.")

    INPUTIMGPATH = args.inputpath 
    DESTROOT_DIR = os.path.join(args.savepath, "Hybrid_MIGCOW_Attack") 
    DEST_DIR_EVADED = os.path.join(DESTROOT_DIR, "evaded")  
    os.makedirs(DEST_DIR_EVADED, exist_ok=True)  

    imagelist = [f for f in os.listdir(INPUTIMGPATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    SEMANTIC_ITERATIONS = 50 
    MIGCOW_ITERATIONS = 100 
    
    transform_e4e = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_lpips = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    for xind, imgfile in enumerate(imagelist):
        print(f'\nProcessing: {imgfile}')
        try:
            original_image = Image.open(os.path.join(INPUTIMGPATH, imgfile)).convert("RGB")
            
            if vlm_model is not None and vlm_processor is not None:
                with torch.no_grad():
                    # 1. Get the base neutral description
                    vlm_inputs = vlm_processor(original_image, return_tensors="pt").to(DEVICE)
                    out = vlm_model.generate(**vlm_inputs, max_new_tokens=50)
                    base_desc = vlm_processor.decode(out[0], skip_special_tokens=True)
                    
                    # 2. Force the VLM to generate a stylistic modifier
                    # We pass the base description back in and ask the VLM to continue it 
                    # with photographic/stylistic details.
                    conditional_prompt = f"A photograph of {base_desc}, featuring"
                    vlm_modifier_inputs = vlm_processor(original_image, text=conditional_prompt, return_tensors="pt").to(DEVICE)
                    out_mod = vlm_model.generate(**vlm_modifier_inputs, max_new_tokens=20)
                    extended_desc = vlm_processor.decode(out_mod[0], skip_special_tokens=True)
                    
                neutral_txt = base_desc
                # The target text is now dynamically generated based on the image's exact context
                target_txt = extended_desc 
                print(f"  [VLM Prompt] Neutral: '{neutral_txt}' | Target: '{target_txt}'")
            else:
                # Fallback to static list if VLM fails
                rindex = random.randint(0, len(allneutrals)-1)  
                neutral_txt, target_txt = allneutrals[rindex], alltargets[rindex]

            dt = GetDt([target_txt, neutral_txt], model).detach()

            with torch.no_grad():
                img_ref_clip = clip_transforms1(original_image).unsqueeze(0).to(DEVICE)
                img_ref_clip = clip_transforms2(img_ref_clip)
                src_img_features = model.encode_image(img_ref_clip)
                src_img_features = src_img_features / src_img_features.norm(dim=-1, keepdim=True)

           
            # =========================================================
            # FIND CLOSEST REAL IMAGE MATCH & EXTRACT 2 TARGETS
            # =========================================================
            with torch.no_grad():
                sims = torch.cosine_similarity(src_img_features, real_embeddings_tensor)
                best_idx = sims.argmax().item()
                best_real_path = real_paths[best_idx]
                
                real_img_pil = Image.open(best_real_path).convert("RGB")
                real_tensor_01 = transforms.ToTensor()(real_img_pil).unsqueeze(0).to(DEVICE)
                real_tensor_01 = F.interpolate(real_tensor_01, size=(1024, 1024), mode='bicubic', align_corners=False)
                
                # TARGET 1: Raw Image Radial Stats (For Det 1 & 3)
                target_stats_radial = extract_radial_dct_stats(real_tensor_01).detach()
                
                # TARGET 2: Residual Radial Stats (For Det 2)
                down_real = F.interpolate(real_tensor_01, scale_factor=0.25, mode='area')
                up_real = F.interpolate(down_real, size=(1024, 1024), mode='bicubic', align_corners=False)
                real_residual = torch.abs(real_tensor_01 - up_real)
                target_stats_res_radial = extract_radial_dct_stats(real_residual).detach()

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
            # PHASE 1: SEMANTIC OPTIMIZATION (Unchanged)
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
            # UPDATED PHASE 2: UNIFIED 3-OBJECTIVE EVASION
            # =========================================================
            print("  -> Executing Unified Radial Frequency & CNN Evasion Attack...")
            with torch.no_grad():
                img_gen_base_raw, _ = g_ema([latent_in], input_is_latent=True, noise=noises, boundary_tmp2=boundary_curr, alpha=current_alpha, use_dt=True)
                if img_gen_base_raw.shape[0] == g_ema.n_latent:
                    img_gen_base_raw = img_gen_base_raw[-1].unsqueeze(0)

            base_detached = img_gen_base_raw.detach()
            delta = torch.zeros(1, 3, 1024, 1024, device=DEVICE)
            g_momentum = torch.zeros_like(delta)

            num_patches = 4
            patch_size = 224
            patches_coords = [(random.randint(0, 1024 - patch_size), random.randint(0, 1024 - patch_size)) for _ in range(num_patches)]

            # --- SPATIAL EVASION LOSSES ---
            def tv_loss(d_in):
                """ Ensures noise remains continuous (evades Det 2/3 CNNs) """
                tv_h = torch.sum(torch.abs(d_in[:, :, 1:, :] - d_in[:, :, :-1, :]))
                tv_w = torch.sum(torch.abs(d_in[:, :, :, 1:] - d_in[:, :, :, :-1]))
                return (tv_h + tv_w) / (d_in.shape[2] * d_in.shape[3])
              
            def scrambler_loss(d_in, block_size=8):
                """ Breaks up rigid grid formations (evades Det 2 residual CNN) """
                b, c, h, w = d_in.shape
                blocks = d_in.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
                blocks = blocks.contiguous().view(b, c, -1, block_size * block_size)
                block_vars = torch.var(blocks, dim=-1)
                var_of_vars = torch.var(block_vars, dim=-1).mean()
                return -var_of_vars

            def loss_fn_cls(d_in):
                """ Spatial Objective: Semantics + Dual CNN Evasion """
                img_final = base_detached + d_in
                total_loss = 0
                for (h, w) in patches_coords:
                    x_patch = img_final[:, :, h:h+patch_size, w:w+patch_size]
                    x_cls = F.interpolate(x_patch, size=(448, 448), mode='bilinear', align_corners=False)
                    x_cls_norm = clip_transforms2(to_01(x_cls))
                    total_loss += criterion(classifier(x_cls_norm), torch.tensor([0], device=DEVICE))
                
                # Combine TV (smoothness) and Scrambler (anti-grid)
                return (total_loss / num_patches) + (100.0 * tv_loss(d_in)) + (100.0 * scrambler_loss(d_in))

            # --- FREQUENCY EVASION LOSSES ---
            def loss_fn_freq_radial(d_in):
                """ Frequency Objective 1: Fools Det 1 and Det 3 """
                img_final = to_01(base_detached + d_in)
                stats_adv = extract_radial_dct_stats(img_final)
                return 100.0 * F.mse_loss(stats_adv, target_stats_radial)
                
            def loss_fn_freq_res(d_in):
                """ Frequency Objective 2: Fools Det 2 """
                img_final = to_01(base_detached + d_in)
                downscaled = F.interpolate(img_final, scale_factor=0.25, mode='area')
                upscaled = F.interpolate(downscaled, size=(1024, 1024), mode='bicubic', align_corners=False)
                residual = torch.abs(img_final - upscaled)
                stats_adv = extract_radial_dct_stats(residual)
                return 100.0 * F.mse_loss(stats_adv, target_stats_res_radial)

            # ---------------------------------------------------------
            # 3. THE MIG-COW OPTIMIZATION LOOP
            # ---------------------------------------------------------
            base_name = os.path.splitext(imgfile)[0]

            for tempi in range(MIGCOW_ITERATIONS):
                delta_adv = delta.detach().requires_grad_(True)
                baseline_delta = torch.zeros_like(delta_adv)
                
                # Generate 3 precise orthogonal vectors
                g_cls = integrated_gradients(delta_adv, baseline_delta, loss_fn_cls, steps=args.ig_steps)
                g_rad = integrated_gradients(delta_adv, baseline_delta, loss_fn_freq_radial, steps=args.ig_steps)
                g_res = integrated_gradients(delta_adv, baseline_delta, loss_fn_freq_res, steps=args.ig_steps)
                
                # Combine all 3 objectives safely
                g_cow = compute_cow_gradient([g_cls, g_rad, g_res], beta=args.cow_beta)
                
                l1_norm = torch.norm(g_cow, p=1) + 1e-8
                g_momentum = args.mu * g_momentum + (g_cow / l1_norm)
                
                with torch.no_grad():
                    delta = delta - args.adv_step * torch.sign(g_momentum)
                    delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)

                # ========================================================
                # REAL-TIME DEBUGGING: Save intermediate steps & Monitor ALL objectives
                # ========================================================
                if tempi % 20 == 0:  
                    with torch.no_grad():
                        img_current = to_01(base_detached + delta)
                        
                        # 1. Spatial constraints
                        current_tv = tv_loss(delta).item() * 100.0  
                        current_scram = scrambler_loss(delta).item() * 100.0
                        
                        # 2. Frequency constraints (MSE against target real stats)
                        current_rad_mse = F.mse_loss(extract_radial_dct_stats(img_current), target_stats_radial).item() * 100.0
                        
                        down_curr = F.interpolate(img_current, scale_factor=0.25, mode='area')
                        up_curr = F.interpolate(down_curr, size=(1024, 1024), mode='bicubic', align_corners=False)
                        current_res_mse = F.mse_loss(extract_radial_dct_stats(torch.abs(img_current - up_curr)), target_stats_res_radial).item() * 100.0
                        
                        print(f"  [Iter {tempi}] TV: {current_tv:.2f} | Scram: {current_scram:.2f} | Rad MSE: {current_rad_mse:.4f} | Res MSE: {current_res_mse:.4f}")
                        
                        delta_vis = (delta[0].cpu().permute(1, 2, 0).numpy() / args.epsilon)
                        delta_vis = ((delta_vis + 1) / 2 * 255).astype(np.uint8)
                        
                        debug_dir = os.path.join(DESTROOT_DIR, "debug_steps")
                        os.makedirs(debug_dir, exist_ok=True)
                        Image.fromarray(delta_vis).save(os.path.join(debug_dir, f"noise_{base_name}_iter{tempi}.png"))

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







