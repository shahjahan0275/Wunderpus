import os
import sys
import torch

# -----------------------------
# STEP 1: Set CUDA_VISIBLE_DEVICES correctly
# Use numeric index for MIG slice, NOT full UUID
# If your MIG slice is the first MIG device, use "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# -----------------------------

# -----------------------------
# STEP 2: Force PyTorch CUDA init
# -----------------------------
torch.cuda.init()  # ensures CUDA runtime is ready

# -----------------------------
# STEP 3: Set DEVICE safely
# -----------------------------
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# -----------------------------
# STEP 4: Debug info
# -----------------------------
print(f"--- DEBUG INFO ---")
print(f"Torch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 0:
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Target Device: {DEVICE}")

# -----------------------------
# STEP 5: Paths for testing
# -----------------------------
e4e_path = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/encoder4editing/e4e_ffhq_encode.pt"
surrogate_path = '/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth'

# -----------------------------
# STEP 6: Function to load checkpoints
# -----------------------------
def check_load(path, name):
    print(f"\nAttempting to load {name}...")
    try:
        # Load to CPU first
        ckpt = torch.load(path, map_location='cpu')
        print(f"✅ Loaded {name} to CPU...")

        # Optional: just a test, no need to move weights now
        print(f"✅ SUCCESS: {name} is valid.")
        return True
    except Exception as e:
        print(f"❌ FAIL: {name} error: {e}")
        return False

# -----------------------------
# STEP 7: Run checks
# -----------------------------
res1 = check_load(e4e_path, "e4e Encoder")
res2 = check_load(surrogate_path, "Surrogate Classifier")

# -----------------------------
# STEP 8: Exit codes for tcsh script
# -----------------------------
if not (res1 and res2):
    sys.exit(1)
else:
    print("Final Status: SUCCESS")
    sys.exit(0)

