#!/bin/tcsh

##################################################
# 1. Clean environment + load correct CUDA
##################################################
module purge
module load cuda/11.5

##################################################
# 2. MIG-safe CUDA environment (CRITICAL)
##################################################
# SLURM exports MIG UUID → PyTorch 1.13 breaks
# Force numeric mapping so MIG slice becomes cuda:0
setenv CUDA_VISIBLE_DEVICES 0
setenv NVIDIA_VISIBLE_DEVICES 0

setenv CCCL_IGNORE_DEPRECATED_CPP_DIALECT 1
setenv TORCH_CUDA_ARCH_LIST "8.0"

##################################################
# 3. PYTHONPATH (tcsh-safe)
##################################################
set STYLECLIP_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack/stylegan2-pytorch/StyleCLIP/global_directions"

if ( $?PYTHONPATH ) then
    setenv PYTHONPATH "${STYLECLIP_DIR}:${PYTHONPATH}"
else
    setenv PYTHONPATH "${STYLECLIP_DIR}"
endif

##################################################
# 4. Clear torch extension cache (important)
##################################################
rm -rf ~/.cache/torch_extensions/fused_v1
rm -rf ~/.cache/torch_extensions/upfirdn2d_v1

##################################################
# 5. CUDA sanity check (FAIL FAST)
##################################################
echo "---- CUDA SANITY CHECK ----"
python - << EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("Device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("❌ CUDA NOT VISIBLE TO PYTORCH")
EOF

if ( $status != 0 ) then
    echo "❌ CUDA CHECK FAILED — aborting."
    exit 1
endif

##################################################
# 6. Model load test
##################################################
echo "--- Step 1: Running hardware and model load test ---"
python test_load.py

if ( $status != 0 ) then
    echo "❌ TEST FAILED: Stopping here to save time."
    exit 1
endif

echo "✅ Test passed! Starting main attack..."

##################################################
# 7. Run attack
##################################################
# For StyleGAN + MIG_COW

#python adversarialattack_clipresnet_stylegan_FOS_MIG_COW_CAMME_WB_SDVAE_Ablation_MCobjectives_Phase2.py \
#python adversarialattack_clipresnet_stylegan_FOS_MIG_COW_CAMME_WB_SDVAE_Ablation_Phase1_2_LPIPS_CAM.py \
#python adversarialattack_clipresnet_stylegan_FOS_MIG_COW_CAMME_WB_SDVAE_Ablation_Phase1_2.py \
python adversarialattack_clipresnet_stylegan_FOS_MIG_COW_CAMME_WB_SDVAE_Ablation_Phase1_2_Spatial_DCT_4thOrd.py \
    #--inputpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/fake_test" \
    #--inputpath "/speed-scratch/a_shahj/D3/data/MIG-COW/test/fake/hq" \
    --inputpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/SD_dataset/test/fake" \
    #--inputpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/provenance/images/ft-sd21-pcb-100k-85476548432385702seed" \
    #--inputpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/provenance/images/sd21-4567820493seed" \
    --savepath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet_advimage/Double_Phase_Spatial_DCT_4th_D11" \
    #--savepath "/speed-scratch/a_shahj/model_provenance/images" \
    #--realpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real" \
    --realpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/SD_dataset/test/real" \
    --plosscoeff 0.5 \
    --lr 0.005 \
    --alpha 2.5 \
    --beta 0.12 \
    --epsilon 0.1254 \
    --adv_step 0.007843 \
    --ig_steps 7 \
    --cow_beta 0.75 \
    --mu 1.0 \
    --sr-scale 4 

