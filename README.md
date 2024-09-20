# ADP-GFA: Artistic Detail Prediction Network Based on Global Feature Analysis
The winning solution of the ECCV 2024 AI for Visual Arts Saliency Estimation Challenge (AI4VA).

## Note
This project requires two different virtual environments. Please follow the instructions to ensure the code runs smoothly.

### Clone Repo

   ```bash
   git clone --depth 1 https://github.com/Eth138l/ADP-GFA.git
   ```
If you encounter the following error: `error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)`, you can try increasing the Git buffer size with the following command, then retry `git clone`:
   ```bash
   git config --global http.postBuffer 524288000
   ```

## Usage of DeepGaze IIE and MDS-ViTNet
To perform inference with DeepGaze IIE and to train and perform inference with MDS-ViTNet, follow these steps:

### Setting Up Virtual Environment py38
Create Conda Environment and Install Dependencies

   ```bash
   # create new anaconda env
   conda create -n py38 python=3.8 -y
   conda activate py38

   # install python dependencies
   conda install cudatoolkit=11.7
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

### Datasets Preparation
The dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1SRtf8zH_u90cQeT2GgOCUBGcrMk2BUJt?usp=sharing). Please place it in the `data_mds2e` folder and confirm the file structure as follows:
```
ADP-GFA
    |- data_mds2e
       |- images
       |- maps
       |- train_id.csv
       |- val_id.csv
    |- DeepGaze
    |- MDS-ViTNet
    |- results
```

### Pre-trained Weights
The `deepgaze2e.pth` file will be automatically downloaded when the code runs. 

If you encounter network issues, you can download it from [this link](https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth) and move it to the specified directory as indicated in the error message.

The pre-trained model for MDS-ViTNet should be downloaded from [Google Drive](https://drive.google.com/drive/folders/1cTK1J2bNibmzCqoy_dNqPh04GHQYmsKF?usp=drive_link) and placed in the following structure:
```
MDS-ViTNet
    |- checkpoints
        |- best_model_200_2_0.5_1e-4_40.pth
        |- best_model_200_2_0.6_1e-4_40.pth
    |- weights
        |- CNNMerge.pth
        |- ViT_multidecoder.pth
```

## Inference
- Note: The following commands should **not** be modified from their default values.
  
To perform inference with DeepGaze IIE, use the following command:
```bash
cd DeepGaze
python 2e_inference.py --images_path ../data_mds2e/images/test/ --results_path ../results/2e_test_1622 --target_size 1600 2200
```

### Parameters:
- `--img_path`: The path to the folder containing the original images for which you want to generate saliency maps. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--resluts_path`: The path to the folder where you want to save the saliency map results. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--target_size`: The size to which you want to resize the original images.

To perform inference with MDS-ViTNet, use the following command:
```bash
cd MDS-ViTNet
python inference.py --img_path ../data_mds2e/images/test/ --output_dir ../results/MDS_test_trained200_2_0.5_1e-4_40 --path_to_ViT_multidecoder ./checkpoints/best_model_200_2_0.5_1e-4_40.pth
python inference.py --img_path ../data_mds2e/images/test/ --output_dir ../results/MDS_test_trained200_2_0.6_1e-4_40 --path_to_ViT_multidecoder ./checkpoints/best_model_200_2_0.6_1e-4_40.pth
```
- Note: In order to successfully perform model fusion, inference needs to be run using two different pre-trained weight files. Therefore, both of the above lines of code should be executed once.

### Parameters:
- `--img_path`: The path to the folder containing the original images for which you want to generate saliency maps. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--output_dir`: The path to the folder where you want to save the saliency map results. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--path_to_ViT_multidecoder`: The path to the pre-trained weight file you want to use.

## Training
If you want to retrain the MDS-ViTNet model, use the following command:
```bash
python train.py --p 0.5 --path_to_save ./checkpoints
```

- Note: Each training run may exhibit some variability. Executing this command will **overwrite** the locally downloaded pre-trained weight files. To achieve results consistent with those reported, use the downloaded pre-trained weight files for inference without the need for retraining.

### Parameters:
- `--p`: This value determines the probability of data augmentation. Executing the command with the default value will produce the `best_model_200_2_0.5_1e-4_40.pth` file. To retrain and obtain the `best_model_200_2_0.6_1e-4_40.pth` file, simply set the value to 0.6.
- `--path_to_save`: The path where you want to store the training weight files and the final best model weight file.

## Usage of SUM
To train and perform inference with SUM, follow these steps:

### Setting Up Virtual Environment py310

Create Conda Environment and Install Dependencies
- Create and activate the virtual environment:

```
conda create --name py310 python=3.10
conda activate py310
```
- Install PyTorch and other necessary libraries:
```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
### Datasets Preparation
The dataset `AI4VA`can be downloaded from [this link](https://drive.google.com/drive/folders/1_DCOJf0ist3twchYmQBRd_ASUwdqPiRP). Please place it in the `datasets` folder and confirm the file structure as follows:
```
ADP-GFA
    |- SUM-main
        |- datasets
            |- AI4VA
                |- test
                |- train
                |- val
                |- train_id.csv
                |- val_id.csv
    |- DeepGaze
    |- MDS-ViTNet
    |- results
```
### Pre-trained Weights

Download the SUM model from the provided Google Drive link and move it to the specified directory:

- [Download SUM model](https://drive.google.com/drive/folders/1_DCOJf0ist3twchYmQBRd_ASUwdqPiRP): `SUM_newK5.pth`
- Move `SUM_newK5.pth` to: `SUM-main/net/pre_trained_weights`

## Inference
To generate saliency maps, use the `inference.py` script. Here are the steps and commands:
- Note: The following commands should **not** be modified from their default values.

```
cd SUM-main
python test_resize.py
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0471_1954_05_23-14.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0479_1954_07_18-01.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0480_1954_07_25-01.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0485_1954_08_29-01.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0525_1955_06_05-16.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0553_1955_12_18-06.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python inference.py --img_path './datasets/AI4VA/test/test_resize/Vaillant_0608_1957_01_06-06.png' --condition 3 --output_path SUM_newK5 --heat_map_type HOT
python Pixel_conversion.py
python Gaussian_blur.py
```
### Parameters:

- `--img_path`: Path to the input image for which you want to generate the saliency map.
- `--condition`: Condition index for generating the saliency map. Each number corresponds to a specific type of visual content:
  - `0`: Natural scenes based on the Salicon dataset (Mouse data).
  - `1`: Natural scenes (Eye-tracking data).
  - `2`: E-Commercial images.
  - `3`: User Interface (UI) images.
- `--output_path`: Path to the folder where the output saliency map will be saved.
- `--heat_map_type`: Type of heatmap to generate. Choose either `HOT` for a standalone heatmap or `Overlay` to overlay the heatmap on the original image.

## Training

To train the model, first download the necessary pre-trained weights and datasets:
1. **Pretrained Encoder Weights**:Download `vssmsmall_dp03_ckpt_epoch_238.pth` from [Google drive](https://drive.google.com/drive/folders/1_DCOJf0ist3twchYmQBRd_ASUwdqPiRP) and move the file to:

    `./SUM-main/net/checkpoint/vssmsmall_dp03_ckpt_epoch_238.pth.`
3. **Datasets:**

The dataset `AI4VA`can be downloaded from [this link](https://drive.google.com/drive/folders/1_DCOJf0ist3twchYmQBRd_ASUwdqPiRP). Please place it in the `datasets` folder and confirm the file structure as follows:
- Note: If you have already set the dataset format during the inference phase, you **do not need** to set it again during the training phase.

```
ADP-GFA
    |- SUM-main
        |- datasets
            |- AI4VA
                |- test
                |- train
                |- val
                |- train_id.csv
                |- val_id.csv
    |- DeepGaze
    |- MDS-ViTNet
    |- results
```

Run the training process:
```
cd SUM-main
python train_resize.py
python val_resize.py
python FixationMapper_train.py
python FixationMapper_val.py
python train.py
```

## Model Fusion
After completing the previous steps, please check whether your `results` folder has the following structure：
```
results
    |- MDS_test_trained200_2_0.5_1e-4_40
    |- MDS_test_trained200_2_0.6_1e-4_40
    |- SUM_newK5_gaosi
    |- 2e_test_1622
```
If your file structure meets the requirements, run the following code to perform model fusion:
```bash
python tta.py
```
All variables are preset with `default` and should **not** be altered. Upon successful execution, the results will be saved in the `ADP-GFA/results/2MDS_mixSUMk5IIE` directory, which is our final submission.

## Acknowledgment
We would like to thank the authors and contributors of [MDS-ViTNet](https://github.com/ignatpolezhaev/mds-vitnet), [SUM](https://github.com/Arhosseini77/SUM), and [DeepGaze](https://github.com/matthias-k/DeepGaze) for their open-sourced code which significantly aided this project.
