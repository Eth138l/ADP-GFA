# Midsen
The winning solution of the ECCV 2024 AI for Visual Arts Saliency Estimation Challenge (AI4VA).

## Note
This project requires two different virtual environments. Please follow the instructions to ensure the code runs smoothly.

## Usage of DeepGaze IIE and MDS-ViTNet
To perform inference with DeepGaze IIE and to train and perform inference with MDS-ViTNet, follow these steps:

### Setting Up Virtual Environment py38
1. Clone Repo

   ```bash
   git clone https://github.com/Eth138l/Midsen.git
   ```
2. Create Conda Environment and Install Dependencies

   ```bash
   conda create --name py38 --file environment_2e_mds.yml
   ```

- Note: The above command will install `python==3.8.19` and `cuda==11.7`.

### Pre-trained Weights
The `deepgaze2e.pth` file will be automatically downloaded when the code runs. 

If you encounter network issues, you can download it from [this link](https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth) and move it to the specified directory as indicated in the error message.

The pre-trained model for MDS-ViTNet should be downloaded from Google Drive and placed in the following structure:
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
To perform inference with DeepGaze IIE, use the following command:
```bash
python 2e_inference.py --images_path /path/to/your/images --results_path /path/to/output --target_size 1600 2200
```

### Parameters:

- `--img_path`: The path to the folder containing the original images for which you want to generate saliency maps. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--resluts_path`: The path to the folder where you want to save the saliency map results. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--target_size`: The size to which you want to resize the original images, defaulting to 1600x2200.

To perform inference with MDS-ViTNet, use the following command:
```bash
python inference.py --img_path /path/to/your/images --output_dir /path/to/your/results --path_to_ViT_multidecoder ./checkpoints/best_model_200_2_0.5_1e-4_40.pth
```

### Parameters:

- `--img_path`: The path to the folder containing the original images for which you want to generate saliency maps. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--output_dir`: The path to the folder where you want to save the saliency map results. If you encounter a `FileNotFoundError`, it is recommended to use absolute paths.
- `--path_to_ViT_multidecoder`: The path to the pre-trained weight file you want to use, defaulting to `./checkpoints/best_model_200_2_0.5_1e-4_40.pth`. For successful model fusion, you should also use `./checkpoints/best_model_200_2_0.6_1e-4_40.pth` for a second inference, and make sure to save the results in different folders for these two runs.
