# Image-Classifier-Project
A deep learning-based image classifier built using PyTorch. This project includes scripts for training a model on image datasets and making predictions on new images.

Features
✔ Train a model on an image dataset using train.py
✔ Supports multiple architectures from torchvision.models
✔ Customize hyperparameters (learning rate, hidden units, epochs)
✔ Train on GPU or CPU
✔ Make predictions on new images with predict.py
✔ Get top-K predictions with associated probabilities
✔ Load category names from a JSON file for better readability

Installation
Clone this repository:
bash
Copy
Edit
git clone https://github.com/yourusername/Image-Classifier.git
cd Image-Classifier
Install dependencies:
bash
Copy
Edit
pip install torch torchvision matplotlib numpy
Usage
Training the Model
bash
Copy
Edit
python train.py --data_dir path/to/data --arch resnet18 --epochs 10 --learning_rate 0.001 --gpu
Making Predictions
bash
Copy
Edit
python predict.py --image path/to/image.jpg --checkpoint checkpoint.pth --top_k 5 --gpu
Checking Model Performance
To verify if the model is working correctly, the training script prints:
✅ Training Loss
✅ Validation Loss
✅ Validation Accuracy

You can also test predictions using print statements inside predict.py.

Contributing
Feel free to improve this project by submitting a pull request!
