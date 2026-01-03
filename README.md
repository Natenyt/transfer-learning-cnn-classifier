# Transfer Learning - Cats vs. Dogs Classification

A deep learning project implementing transfer learning with **MobileNetV2** pretrained on ImageNet to classify images of cats and dogs. This project demonstrates the power of transfer learning to achieve high accuracy (>94%) on a binary classification task.

## 📋 Project Overview

This project uses transfer learning to fine-tune the MobileNetV2 convolutional neural network for binary classification of cat and dog images. The model leverages pre-trained ImageNet weights and adds a custom classification head.

### Key Features
- ✅ Transfer learning with MobileNetV2
- ✅ Image preprocessing and data augmentation
- ✅ TensorFlow 2.x and Keras API
- ✅ TensorBoard integration for training visualization
- ✅ Achieves >94% accuracy on test set
- ✅ Complete Jupyter notebook implementation

## 🚀 Quick Start

### Option 1: Google Colab (Recommended) ⭐

The easiest way to run this project is using Google Colab with free GPU support:

1. **Open the notebook in Colab:**
   - Upload `lab transfer learning.ipynb` to Google Colab
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. **Upload dataset:**
   - Upload `train.zip` and `test.zip` to the Colab runtime
   - The notebook will automatically unzip the files

3. **Enable GPU:**
   - Go to `Runtime` → `Change runtime type` → Select `GPU`

4. **Run all cells:**
   - Click `Runtime` → `Run all` or run cells sequentially

### Option 2: Local Setup

If you prefer to run locally:

#### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM

#### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Assignment10
   ```

2. **Create virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset:**
   - Ensure `train.zip` and `test.zip` are in the project directory

5. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

6. **Open and run:**
   - Open `lab transfer learning.ipynb`
   - Run all cells sequentially

## 📁 Project Structure

```
Assignment10/
├── lab transfer learning.ipynb  # Main Jupyter notebook
├── lab transfer learning.md     # Lab instructions
├── Lab transfer learning.pdf    # Lab documentation
├── train.zip                     # Training dataset (not in repo)
├── test.zip                      # Test dataset (not in repo)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🗂️ Dataset

The **Cats vs. Dogs** dataset contains:
- **Training set**: ~25,000 images
- **Test set**: ~12,500 images

**Note:** The dataset files (`train.zip` and `test.zip`) are not included in this repository due to their size. You need to obtain them separately.

### Filename Format
Images are named as: `{class}.{number}.jpg`
- Example: `cat.0.jpg`, `dog.1234.jpg`

## 🧠 Model Architecture

1. **Base Model**: MobileNetV2 (pretrained on ImageNet)
   - Input shape: 128×128×3
   - Pretrained weights frozen during training
   
2. **Custom Classification Head**:
   - GlobalAveragePooling2D
   - Dropout (0.2)
   - Dense layer (2 units, softmax activation)

3. **Training Configuration**:
   - Optimizer: Adam (lr=0.001)
   - Loss: Sparse Categorical Crossentropy
   - Metrics: Accuracy
   - Batch size: 32
   - Epochs: 10

## 📊 Results

The model achieves:
- **Test Accuracy**: >94%
- **Training Time**: ~10-15 minutes on GPU
- **Model Size**: Lightweight (MobileNetV2)

## 🔍 What's Inside the Notebook

1. **Data Loading**: Unzip and load training/test datasets
2. **Preprocessing**: Resize images to 128×128 and normalize
3. **Model Building**: Load MobileNetV2 and add classification head
4. **Training**: Train with TensorBoard monitoring
5. **Evaluation**: Test accuracy on full test set
6. **Prediction**: Visualize predictions on sample images

## 📈 TensorBoard

The notebook includes TensorBoard integration for real-time training visualization:
- Loss curves (training & validation)
- Accuracy metrics
- Model graph

TensorBoard will activate automatically during training.

## 🛠️ Technologies Used

- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **Jupyter Notebook**: Interactive development

## 💡 Tips

- **Use GPU**: Training is significantly faster with GPU acceleration
- **Colab is recommended**: Free GPU access and no local setup required
- **Monitor TensorBoard**: Watch training progress in real-time
- **Experiment**: Try different hyperparameters (learning rate, epochs, dropout)

## 📝 Assignment Requirements

This project fulfills the following requirements:
- ✅ Import pretrained MobileNetV2 from ImageNet
- ✅ Remove final classification layer
- ✅ Add custom classification head for 2 classes
- ✅ Create data loader with image resizing (128×128)
- ✅ Use `tf.data.Dataset` for efficient data loading
- ✅ Train with softmax cross-entropy loss
- ✅ Achieve >94% test accuracy

## 🤝 Contributing

This is an academic project. If you'd like to suggest improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is created for educational purposes as part of an AI course assignment.

## 🙋 Questions?

If you encounter any issues:
1. Ensure all dependencies are installed
2. Verify dataset files are uploaded/unzipped correctly
3. Check GPU is enabled in Colab
4. Review error messages in the notebook output

---

**Recommended**: Use Google Colab for the best experience! 🚀
