# Wunderpus
### All the codes required to produce all the datasets are provided here.

## Installation

```bash
conda env create -f environment.yml
conda activate sem_adv
```

If package conflicts occur:

```bash
pip install -r requirements.txt
```

## Pretrained Weights

The pretrained model weights are hosted on Zenodo for permanent archival access.

### Wuderpus_weights

The pretrained weights will be made available through Zenodo at the permanent artifact DOI.

The downloaded archive should be extracted as:

Wuderpus_weights/

Pretrained weights necessary to run the code should be downloaded from the Zenodo link above and placed in the respective folders as follows.

Folder path for the pre-trained weight:

Path for folder "StyleCLIP_global_directions_data_ffhq":   ./StyleCLIP/global_directions/data/ffhq/
          
Path for folder "StyleCLIP_global_torch_npy_ffhq":   ./StyleCLIP/global_torch/npy/ffhq/
          
Path for folder "StyleCLIP_global_torch_npy_human":   ./StyleCLIP/global_torch/npy/human/

Path for folder "encoder4editing":   ./encoder4editing/e4e_ffhq_encode.pt
          
Path for folder "encoder4editing_editings_ganspace_pca":   ./encoder4editing/editings/ganspace_pca/

Path for folder "encoder4editing_editings_interfacegan_directions":   ./encoder4editing/editings/interfacegan_directions/

Path for folder "encoder4editing_pretrained_models":   ./encoder4editing/pretrained_models/

Path for folder "stylegan2-pytorch":   ./stylegan2-pytorch/
         
Path for folder "stylegan2-pytorch_StyleCLIP_global_directions_model":   ./stylegan2-pytorch/StyleCLIP/global_directions/model/

Path for folder "stylegan2-pytorch_StyleCLIP_global_torch_npy_ffhq":   ./stylegan2-pytorch/StyleCLIP/global_directions/npy/ffhq

Path for folder "stylegan2-pytorch_StyleCLIP_global_directions_npy_afhqcat":   ./stylegan2-pytorch/StyleCLIP/global_directions/npy/afhqcat/

Path for folder "stylegan2-pytorch_StyleCLIP_global_directions_npy_afhqdog":   ./stylegan2-pytorch/StyleCLIP/global_directions/npy/afhqdog/

Path for folder "stylegan2-pytorch_StyleCLIP_global_torch_npy_ffhq":   ./stylegan2-pytorch/StyleCLIP/global_torch/npy/ffhq/

Path for folder "stylegan2-pytorch_StyleCLIP_global_torch_npy_human":   ./stylegan2-pytorch/StyleCLIP/global_torch/npy/human/

Path for folder "stylegan2-pytorch_checkpoint":   ./stylegan2-pytorch/checkpoint/

Path for folder "stylegan2-pytorch_lpips_weights_v0.0":   ./stylegan2-pytorch/lpips/weights/v0.0/

Path for folder "stylegan2-pytorch_lpips_weights_v0.1":   ./stylegan2-pytorch/lpips/weights/v0.1/

Path for folder "stylegan2-pytorch_stylegan2-pytorch_editings_ganspace_pca":   ./stylegan2-pytorch/stylegan2-pytorch/editings/ganspace_pca/

Path for folder "stylegan2-pytorch_stylegan2-pytorch_editings_interfacegan_directions":   ./stylegan2-pytorch/stylegan2-pytorch/editings/interfacegan_directions/

## Dataset Preparation

The adversarial examples used in this work are generated from the non-adversarial fake datasets introduced by Abdullah et al.[1] Due to repository size constraints, only representative samples are included from this non-adversarial fake dataset. The original non-adversarial datasets can be obtained from Abdullah et al.[1]. However, we provide sample images sufficient for running the code, though not for full reproducibility of our results. Also we have included only representative samples of all the 19 Attack datasets that we have produced due to the repository size constraints, which should be enough to test the functionality of the code.

To obtain the dataset structure and preprocessing pipeline, we followed the instructions provided in the Evolving Threat repository:

https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect

For the convenience of reproducing the results of our paper we kept our attack scripts in different folder, such as:

1. "GAN-Based_Inversion"  to reproduce the attack of "Table 1"
2. "Diffusion-Based_Inversion" to reproduce the result of "Table 2"
3. "Concept-Specific_Loss_Function_Analysis" to reproduce the result of "Table 3"
4. Both the attack script of step 1 and 2 is required to reproduce the result of "Table 4"
5. "Ablation_Study" to reproduce the result of "Table 5" and "Table 6"

After preparing the dataset according to the instructions of Abdullah et al., please place our attack scripts in the corresponding attack directory:
./adversarialattack/stylegan2-pytorch/

 and execute the commands as follows. For example if you want to run the script to produce the attack dataset D19, the command should be:

```bash
python D19.py
    --inputpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/fake_test" \
    --savepath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet_advimage/MIG_COW_GAN_lpips" \
    --realpath "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real" \
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
```
you should adjust the hyperparameters as described in the paper.

## Acknowledgement

Our code makes references to the following repositories.Also to test our attacks against the dtectors that we used in the paper, please follow the original link of those detectors, given below.

- [AIDE](https://github.com/shilinyan99/AIDE).
- [CAMME] (https://github.com/Magnet300/CAMME).
- [CapsFake] (https://github.com/tuanrpt/CapsFake).
- [CNN-F](https://github.com/PeterWang512/CNNDetection).
- [CNN-F (improved)](https://github.com/shahjahan0275/semantic_adv).
- [CO-SPY](https://github.com/Megum1/Co-Spy).
- [D3](https://github.com/BigAandSmallq/D3).
- [DCT (original)](https://github.com/jonasricker/diffusion-model-deepfake-detection).
- [DCT (improved)](https://github.com/shahjahan0275/semantic_adv).
- [De-Fake](https://github.com/zeyangsha/De-Fake).
- [Directionality](https://github.com/uibk-uncover/directionality).
- [Effort] (https://github.com/YZY-stack/Effort-AIGI-Detection).
- [FakeImageDetection] (https://github.com/chandlerbing65nm/FakeImageDetection).
- [FerretNet] (https://github.com/xigua7105/FerretNet).
- [Resynthesis(original)](https://github.com/SSAW14/BeyondtheSpectrum).
- [Resynthesis (improved)](https://github.com/shahjahan0275/semantic_adv).
- [SAFE] (https://github.com/Ouxiang-Li/SAFE).
- [SPAI] (https://github.com/mever-team/spai)
- [UnivConv2B] (https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect).
- [ViGText] (https://github.com/AhmadALBarqawi/ViGText).

## References

[1] Abdullah, Sifat Muhammad, et al.
    "An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape."
    IEEE Symposium on Security and Privacy (SP), 2024.




