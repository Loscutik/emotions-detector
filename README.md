# 🎭 Emotion Detector
This project implements a real-time facial emotion recognition system using convolutional neural networks (CNN). It covers model building, training, TensorFlow vs. TFLite comparison, and live emotion detection on video streams. Configuration is fully managed via a YAML file for maximum flexibility.

## 🧰 Technologies and Methods

- Python 3.x
- TensorFlow & TFLite for model definition and optimization
- TensorBoard adjusting the training progress
- OpenCV for image capture and frame processing
- Jupyter Notebook for interactive exploration
- PyYAML for configuration loading
- Docker & Docker Compose for environment isolation

## 📁 Project Structure
```
emotions-detector/
├── notebooks/           # Jupyter notebooks for each stage of the workflow
├── scripts/             # Python modules for model building and data preprocessing
├── data/                # CSV datasets and YAML configuration
├── results/             # Training curves, model summaries, comparison plots
├── environments/        # Python virtual environment 
├── Dockerfile           # Instructions to build the Docker image
├── compose.yaml         # Docker Compose setup
└── docker_run.sh        # Helper script to launch the container
```


### 📓 Jupyter Notebooks

| Notebook | Description |
|---|---|
|`train_cnn.ipynb` | Model architecture design and training on the facial emotion dataset|
|`compare_models.ipynb` |  Performance and inference time comparison between TF and TFLite|
|`emotion_detection.ipynb` | Live emotion recognition on webcam  records and webcam feed |

### 🧩 Python Scripts

Script | Purpose
---|---
`scripts/build_model.py` | Defines modular CNN blocks and assembles the full emotion classification model
`scripts/get_data.py` |  Defines functions to get and preprocesse data for training and testing
`scripts/data_handle.py` | Handles image to predict on them: preprocessing (normalization, resizing), prediction functions for TF and TFLite models
`scripts/helpers.py` | Utilities 
`scripts/plotting.py` | Defines functions to work with plots 

### 📊 Data and Configuration

The `data` directory contains:

- train and test datasets; if the datasets are not in the directory, they will  be loaded and unzip from https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip
- `config.yaml` – central configuration file controlling model architecture, training parameters, and compilation settings.

### 📈 Results

All training curves, model summaries, and performance comparison plots are stored in the `results` directory. These artifacts help visualize accuracy and loss trends, as well as inference time differences between TensorFlow and TFLite.

### 🔌 Environment
The project uses Python v3.12.3 with the libraries:
- jupyter
- numpy
- pydantic
- numpy==2.0
- pandas
- matplotlib
- seaborn
- tabulate
- scikit-learn==1.5.1
- tensorflow==2.19.0
- keras==3.6.0
- tf_keras
- opencv-python==4.10.0.84
- pyyaml


### 🐳 Docker
Lets run the project in a Docker container.


## 🏗️ Model Creation

The CNN model is built from reusable functional blocks. Each block is implemented as a separate function for clarity and modularity. The exact architecture is defined in config.yaml.


### 🧠 Key Functions in `build_model.py`

| Function | Purpose |
|---------|--------|
| `data_augmenter()` | Builds an image augmentation pipeline using `ImageDataGenerator`. Includes horizontal flips, shifts, zooms, and rotations. |
| `bottleneck_block(x, filters)` | Implements a MobileNet-style bottleneck block: `Conv2D → BatchNorm → Dropout → ReLU → DepthwiseConv2D → BatchNorm → ReLU → Conv2D → BatchNorm → Dropout → Add`. |
| `create_MobNetLike(input_shape, num_classes)` | Assembles a lightweight CNN model inspired by MobileNet, using `bottleneck_block`-s, `GlobalAveragePooling2D` and `Dense` layers. |
| `identity_block2(x, filters)` | Constructs a ResNet-style identity block: `(Conv2D → BatchNorm → Dropout ) x 2 → ReLU → Add`. |
| `pooling_block(x)` | Adds to identity_block components  a layer that decreases the output size (height and width). `(Conv2D → BatchNorm → Dropout ) x 2 → Conv2D(changing size) → BatchNorm → Dropout → ReLU → Add` |
| `create_ResNet2(input_shape, num_classes)` | Builds a CNN architecture resembling ResNet using identity and pooling blocks , `GlobalAveragePooling2D`, and `Dense` layers. |


### 🧩 Key Classes in `build_model.py`

| Class | Purpose |
|-------|---------|
| `BlockNumerator` | A utility class that assigns unique names to model blocks by maintaining a counter. Useful for clean layer naming. |
| `ConfusingMatrixLog` | Logs a confusion matrix at each training epoch. Saves matrices in `.npy` and `.png` formats to facilitate post-training analysis. |


### 🔧 Implementation Features
- The architectures (create_MobNetLike, create_ResNet2) are built using modular blocks, enabling flexible experimentation.
- The BlockNumerator and ConfusingMatrixLog classes improve debugging and model monitoring.
- All model-building logic is self-contained.
- Each functional block is clearly defined and reusable, which boosts readability and maintainability.
- The normalization parametr lets include BatchNormalization layers to stabilize and accelerate training.
- Dropouts parrameters lets include Dropout layers to reduce overfitting risk.

## Running 
If you use Python virtual environment, you can set up the one from a file in the `environments` directory.
If you don't work with virtual environments or do not want to, you can use Docker image.
Or you can just read html version of the notebooks which you can find in results/html_notebook directory. 

### Virtual environment 

1. Create an identical environment

   - using conda 
       - (creates an environment named 'sklearn'):
       `conda env create -f environment.yml`
       - activate the environment:
       `conda activate sklearn`
   OR
   - using pip:
     `pip install -r requirements.txt`

2. Run the notebook.

  `jupyter lab notebook/nlp_eneriched_news.ipynb`

### Docker
Run a script:
- on Windows:    `docker_run.bat`
- on Linux:      `docker_run.sh`
    
- If you use WSL on Windows 11, you can encounter the error "UtilAcceptVsock:250: accept4 failed 110" (I did :) ).
In this case run 
  - `pre_docker_run_4WSL_win11.sh`
   and then
  - `docker_run.sh`
    
If it is needed, give execute permissions to the scripts:
- `chmod +x pre_docker_run_4WSL_win11.sh`
- `chmod +x docker_run.sh`

After the container runs, the script will show url to open jupyter in your browser. 
If it do not, in another terminal run `docker exec c-emotions jupyter server list` to see the link.

In Jupyter on the left side you will see files list.
You can find the notebooks in `notebook` directory.


## kood-task's links
[subject] (https://github.com/01-edu/public/tree/master/subjects/ai/emotions-detector)  
[audit] (https://github.com/01-edu/public/tree/master/subjects/ai/emotions-detector/audit)

## Author

Olena Budarahina ([obudarah](https://01.kood.tech/git/obudarah))  
Feedback and contributions are welcome! Please open an issue or submit a pull request if you find bugs or have ideas for enhancement.