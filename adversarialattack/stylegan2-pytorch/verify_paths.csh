#!/bin/tcsh

# --- Define Paths ---
set PROJECT_ROOT = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/adversarialattack"
set E4E_PT = "$PROJECT_ROOT/encoder4editing/e4e_ffhq_encode.pt"
set FS3_NPY = "$PROJECT_ROOT/stylegan2-pytorch/StyleCLIP/global_directions/npy/ffhq/fs3.npy"
set G_PT = "$PROJECT_ROOT/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt"
set CLASSIFIER_PT = "$PROJECT_ROOT/stylegan2-pytorch/checkpoint/surrogate_CLIPResNet.pth"

echo "--------------------------------------------------"
echo "🚀 Starting Pre-flight Path Verification..."
echo "--------------------------------------------------"

# Function-like check for directories
foreach dir ($PROJECT_ROOT)
    if ( -d $dir ) then
        echo "✅ Found Directory: $dir"
    else
        echo "❌ MISSING Directory: $dir"
    endif
end

# Check Critical Files
echo "Checking Files..."

if ( -e $E4E_PT ) then
    echo "✅ Found e4e Encoder: $E4E_PT"
else
    echo "❌ MISSING e4e Encoder: $E4E_PT"
endif

if ( -e $FS3_NPY ) then
    echo "✅ Found StyleCLIP Matrix: $FS3_NPY"
else
    echo "❌ MISSING StyleCLIP Matrix: $FS3_NPY"
endif

if ( -e $G_PT ) then
    echo "✅ Found StyleGAN Checkpoint: $G_PT"
else
    echo "❌ MISSING StyleGAN Checkpoint: $G_PT"
endif

if ( -e $CLASSIFIER_PT ) then
    echo "✅ Found Surrogate Classifier: $CLASSIFIER_PT"
else
    echo "❌ MISSING Surrogate Classifier: $CLASSIFIER_PT"
endif

echo "--------------------------------------------------"
echo "Pre-flight check complete."
