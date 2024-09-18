# ADP-GFA: Artistic Detail Prediction Network Based on Global Feature Analysis
The winning solution of the ECCV 2024 AI for Visual Arts Saliency Estimation Challenge (AI4VA).

## Note
This project requires two different virtual environments. Please follow the instructions to ensure the code runs smoothly.

### Clone Repo

   ```bash
   git clone https://github.com/Eth138l/ADP-GFA.git
   ```

## Usage of DeepGaze IIE and MDS-ViTNet
To perform inference with DeepGaze IIE and to train and perform inference with MDS-ViTNet, follow these steps:

### Setting Up Virtual Environment py38
Create Conda Environment and Install Dependencies

   ```bash
   conda create --name py38 --file environment_2e_mds.yml
   ```

- Note: The above command will install `python==3.8.19` and `cuda==11.7`.

### Datasets Preparation
The dataset for the current step is located in the `data_mds2e` folder. Please confirm the file structure as follows:
```
ADP-GFA
    |- data_mds2e
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
python 2e_inference.py --images_path ../data/images/test/ --results_path ../results/2e_test_1622 --target_size 1600 2200
```

### Parameters:
- `--img_path`: The path to the folder containing the original images for which you want to generate saliency maps. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--resluts_path`: The path to the folder where you want to save the saliency map results. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--target_size`: The size to which you want to resize the original images.

To perform inference with MDS-ViTNet, use the following command:
```bash
python inference.py --img_path ../data/images/test/ --output_dir ../results/MDS_test_trained200_2_0.5_1e-4_40 --path_to_ViT_multidecoder ./checkpoints/best_model_200_2_0.5_1e-4_40.pth
python inference.py --img_path ../data/images/test/ --output_dir ../results/MDS_test_trained200_2_0.6_1e-4_40 --path_to_ViT_multidecoder ./checkpoints/best_model_200_2_0.6_1e-4_40.pth
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
Todo

### Datasets Preparation
The dataset for the current step is located in the `Todo` folder. Please confirm the file structure as follows:
```
Todo
```

### Pre-trained Weights
Todo

## Inference
Todo

### Parameters:
Todo

## Training
Todo

### Parameters:
Todo

## Model Fusion
After completing the previous steps, please check whether your `results` folder has the following structure：
```
results
    |- MDS_test_trained200_2_0.5_1e-4_40
    |- MDS_test_trained200_2_0.6_1e-4_40
    |- SUM_newK5_heibai_gaosi_31
    |- 2e_test_1622
```
If your file structure meets the requirements, run the following code to perform model fusion:
```bash
python tta.py
```
All variables are preset with `default` and should **not** be altered. Upon successful execution, the results will be saved in the `ADP-GFA/results/2MDS_mixSUMk5IIE` directory, which is our final submission.

## Acknowledgment
We would like to thank the authors and contributors of [MDS-ViTNet](https://github.com/ignatpolezhaev/mds-vitnet), [SUM](https://github.com/Arhosseini77/SUM), and [DeepGaze](https://github.com/matthias-k/DeepGaze) for their open-sourced code which significantly aided this project.
