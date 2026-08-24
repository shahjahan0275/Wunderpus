import argparse
import os
import sys
import math
import torch 
from torch.nn import functional as F 
from PIL import Image
from tqdm import tqdm 
import torch.nn as nn 
import torchvision.transforms as transforms
import lpips 
import numpy as np 
import clip 
import pandas as pd
import random  
import torch_dct as dct
import traceback

# --- FIX FOR LPIPS IMPORT CONFLICT ---
if hasattr(lpips, '__file__') and lpips.__file__ and 'site-packages' not in os.path.abspath(lpips.__file__):
    if not hasattr(lpips, 'LPIPS'):
        original_sys_path = sys.path[:]
        if hasattr(lpips, '__path__'):
             bad_dir = os.path.dirname(os.path.dirname(os.path.abspath(lpips.__file__)))
        else:
             bad_dir = os.path.dirname(os.path.abspath(lpips.__file__))
        sys.path = [p for p in sys.path if os.path.abspath(p) != bad_dir]
        if 'lpips' in sys.modules: del sys.modules['lpips']
        try:
            import lpips
        except ImportError:
            pass
        finally:
            sys.path = original_sys_path

# --- CONFIGURATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.benchmark = True

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_image(tensor, path):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor.clamp(-1, 1) + 1) / 2.0
    img = transforms.ToPILImage()(tensor)
    img.save(path)

# --- MIG-COW ALGORITHMS ---
def integrated_gradients(x, baseline, loss_fn, steps=5):
    """Computes Integrated Gradients along a linear path from baseline to x."""
    x.requires_grad_(True)
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]
    
    total_gradients = torch.zeros_like(x)
    for scaled_x in scaled_inputs:
        scaled_x = scaled_x.detach().requires_grad_(True)
        loss = loss_fn(scaled_x)
        grad = torch.autograd.grad(loss, scaled_x)[0]
        total_gradients += grad
        
    avg_grad = total_gradients / steps
    integrated_grad = (x - baseline) * avg_grad
    return integrated_grad.detach()

def compute_cow_gradient(grads, beta=0.75, delta=1e-8):
    """Consensus-Orthogonal Weighting (COW) from the MIG-COW paper."""
    N = len(grads)
    norm_grads = [g / (g.norm(p=2) + 1e-8) for g in grads]
    g_con = sum(norm_grads) / N

    flattened_grads = [g.view(-1) for g in norm_grads]
    G = torch.stack(flattened_grads, dim=1) 
    K = torch.matmul(G.t(), G)              

    # RIDGE REGULARIZATION: Prevents linalg.eigh convergence crashes
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

# --- MODELS ---
class OpenAICLIPModel(nn.Module):
    def __init__(self, modelsize, num_labels=2):
        super(OpenAICLIPModel, self).__init__()
        if modelsize=="xxl":
            self.model, _ = clip.load("RN50x64", device=DEVICE)
            self.emb_dim = 1024
        elif modelsize=="base":
            self.model, _ = clip.load("ViT-B/32", device=DEVICE)
            self.emb_dim = 512
        self.fc1 = nn.Linear(self.emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        with torch.amp.autocast('cuda'):
            image_features = self.model.encode_image(inputs).float()
            out = self.relu(self.fc1(image_features)) 
            out = self.relu(self.fc2(out)) 
            out = self.fc3(out)
        return out 

if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--savepath", type=str, default="results")     
    parser.add_argument("--inputpath", type=str, required=True)
    
    parser.add_argument("--epsilon", type=float, default=32.0/255.0, help="Max perturbation bound")
    parser.add_argument("--alpha", type=float, default=4.0/255.0, help="Step size for pixel update")
    parser.add_argument("--steps", type=int, default=30, help="Number of attack iterations (T)")
    parser.add_argument("--ig_steps", type=int, default=5, help="Steps for Integrated Gradient approx")
    
    parser.add_argument("--cow_beta", type=float, default=0.75, help="Weight for consensus gradient")
    parser.add_argument("--mu", type=float, default=1.0, help="Momentum decay factor")
    args = parser.parse_args() 

    try:
        classifier = OpenAICLIPModel("xxl").to(DEVICE).eval()
        classifier.load_state_dict(torch.load('/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth'))
        criterion = nn.CrossEntropyLoss()
    except Exception as e:
        print("\n❌ [ERROR] Failed to load Classifier:")
        traceback.print_exc()
        sys.exit(1)

    DEST_DIR_EVADED = os.path.join(args.savepath, "MIG_COW_Attack", "evaded")  
    os.makedirs(DEST_DIR_EVADED, exist_ok=True)  

    imagelist = [f for f in os.listdir(args.inputpath) if f.endswith(('.png', '.jpg'))]

    for imgfile in tqdm(imagelist, desc="Processing Images"):
        torch.cuda.empty_cache()

        try:
            img_path = os.path.join(args.inputpath, imgfile)
            img_tensor = transforms.Compose([
                transforms.Resize((512, 512)), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            
            x_adv = img_tensor.clone().detach().to(DEVICE)
            baseline = -torch.ones_like(x_adv) 
            g_momentum = torch.zeros_like(x_adv)

            for t in range(args.steps):
                x_adv.requires_grad_(True)
                
                # --- 1. Define ENSEMBLE Objective Functions ---
                
                def loss_fn_global(x_in):
                    x_cls = F.interpolate(x_in, size=(448, 448), mode='bilinear', align_corners=False)
                    x_cls_norm = transforms.Normalize([0.4814, 0.4578, 0.4082], [0.2686, 0.2613, 0.2757])((x_cls.clamp(-1, 1) + 1) / 2.0)
                    return criterion(classifier(x_cls_norm), torch.tensor([1], device=DEVICE))
                
                num_patches = 4
                patch_size = 224
                patches_coords = [(random.randint(0, x_adv.shape[2] - patch_size), 
                                   random.randint(0, x_adv.shape[3] - patch_size)) for _ in range(num_patches)]
                
                def loss_fn_patches(x_in):
                    total_loss = 0
                    for (h, w) in patches_coords:
                        x_patch = x_in[:, :, h:h+patch_size, w:w+patch_size]
                        x_cls = F.interpolate(x_patch, size=(448, 448), mode='bilinear', align_corners=False)
                        x_cls_norm = transforms.Normalize([0.4814, 0.4578, 0.4082], [0.2686, 0.2613, 0.2757])((x_cls.clamp(-1, 1) + 1) / 2.0)
                        total_loss += criterion(classifier(x_cls_norm), torch.tensor([1], device=DEVICE))
                    return total_loss / num_patches

                # =========================================================================
                # UPDATED: Generalized Unconstrained Suppression Attack
                # =========================================================================
                def loss_fn_freq(x_in):
                    # 1. Spatial TV Loss
                    # Defeats SR-residual models by keeping noise naturally smooth
                    tv_h = torch.sum(torch.abs(x_in[:, :, 1:, :] - x_in[:, :, :-1, :]))
                    tv_w = torch.sum(torch.abs(x_in[:, :, :, 1:] - x_in[:, :, :, :-1]))
                    loss_tv = (tv_h + tv_w) / (x_in.shape[2] * x_in.shape[3])

                    x_norm = (x_in.clamp(-1, 1) + 1) / 2.0
                    gray = 0.299 * x_norm[:, 0:1, :, :] + 0.587 * x_norm[:, 1:2, :, :] + 0.114 * x_norm[:, 2:3, :, :]
                    
                    # 2. Global High-Frequency Energy Suppression
                    # Targets the core anomaly (spikes in mid/high bands) rather than forcing rigid distribution shapes
                    d_global = torch.abs(dct.dct_2d(gray.squeeze(0))) # [1, 512, 512]
                    H_d, W_d = d_global.shape[-2], d_global.shape[-1]
                    yy, xx = torch.meshgrid(torch.arange(H_d, device=DEVICE), torch.arange(W_d, device=DEVICE), indexing="ij")
                    cy, cx = (H_d - 1) / 2, (W_d - 1) / 2
                    
                    dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2) / math.sqrt(cy**2 + cx**2)
                    mid_high_mask = (dist > 0.25).float().unsqueeze(0)
                    loss_hf = (d_global * mid_high_mask).mean()

                    # 3. Local Spatial AC Variance & Kurtosis Minimization
                    # Squashes the heavy-tails of the DCT coefficients to evade standard JPEG-based 4th order statistics
                    b, c, h, w = gray.shape
                    hc, wc = (h // 8) * 8, (w // 8) * 8
                    spatial_blocks = gray[:, :, :hc, :wc].unfold(2, 8, 8).unfold(3, 8, 8).contiguous().reshape(b, c, -1, 8, 8)
                    local_d = torch.abs(dct.dct_2d(spatial_blocks)).reshape(b, c, -1, 64) 
                    
                    ac_coeffs = local_d[:, :, :, 1:] # Shape: [B, C, Num_Blocks, 63]
                    
                    # Fixed Broadcasting: keepdim=True ensures tensors align for Z-score calculation
                    var_l = ac_coeffs.var(dim=2, keepdim=True) 
                    s_l = torch.sqrt(var_l + 1e-8)
                    z_l = (ac_coeffs - ac_coeffs.mean(dim=2, keepdim=True)) / s_l
                    
                    kurt_l = (z_l ** 4).mean(dim=2)
                    
                    # Directly minimize the magnitude of Variance and Kurtosis (squash the tail)
                    loss_local_stats = var_l.mean() + kurt_l.mean()

                    return loss_tv + (0.5 * loss_hf) + (0.01 * loss_local_stats)
                # =========================================================================

                # --- 2. Compute Integrated Gradients for the Ensemble ---
                g_global = integrated_gradients(x_adv, baseline, loss_fn_global, steps=args.ig_steps)
                g_patches = integrated_gradients(x_adv, baseline, loss_fn_patches, steps=args.ig_steps)
                g_freq = integrated_gradients(x_adv, baseline, loss_fn_freq, steps=args.ig_steps)

                # --- 3. MIG-COW: Consensus-Orthogonal Weighting ---
                
                g_cow = compute_cow_gradient([g_global, g_patches, g_freq], beta=args.cow_beta)
                
                # --- 4. Momentum Update ---
                l1_norm = torch.norm(g_cow, p=1) + 1e-8
                g_momentum = args.mu * g_momentum + (g_cow / l1_norm)
                
                # --- 5. Pixel Update and Epsilon Clipping ---
                with torch.no_grad():
                    # Full RGB manipulation
                    x_adv = x_adv - args.alpha * torch.sign(g_momentum)
                    
                    # Strict visual bounding
                    perturbation = torch.clamp(x_adv - img_tensor, min=-args.epsilon, max=args.epsilon)
                    x_adv = torch.clamp(img_tensor + perturbation, min=-1.0, max=1.0)

            save_image(x_adv, os.path.join(DEST_DIR_EVADED, f"evaded_{imgfile}"))

        except Exception as e:
            print(f"\n❌ Error processing image {imgfile}:")
            traceback.print_exc()
            continue