This project is a deep learning-based image classifier built using PyTorch. It trains a neural network on an image dataset and makes predictions on new images. The model can be trained on CPU or GPU and supports multiple architectures from torchvision.models.

Features
  Train a deep learning model on image datasets
 
  Choose from multiple pre-trained architectures (e.g., VGG16)
 
  Customize hyperparameters (learning rate, hidden units, epochs)
 
  Train on GPU or CPU for faster computation
 
  Predict the class of an image with predict.py
 
  Display top-K predictions with confidence scores
 
  Load category names from a JSON file for easy interpretation
 

Setup & Installation

Clone the repository:
git clone https://github.com/biohackednidix/Image-Classifier-Project.git

cd Image-Classifier

Install dependencies:
pip install torch torchvision matplotlib numpy

How to Use
1️ Train the Model
Use train.py to train the image classifier.
python train.py --data_dir path/to/data --arch resnet18 --epochs 10 --learning_rate 0.001 --gpu


🔹 Arguments:

--data_dir → Path to dataset

--arch → Model architecture (e.g. vgg16)

--epochs → Number of training epochs 

--learning_rate → Learning rate for optimization

--gpu → Train using GPU 

2️ Make Predictions

Use predict.py to classify an image.

python predict.py --image path/to/image.jpg --checkpoint checkpoint.pth --top_k 5 --gpu

🔹 Arguments:

--image → Path to input image
--checkpoint → Saved model checkpoint
--top_k → Number of top predictions (default: 5)
--gpu → Use GPU for prediction (optional)

How to Verify if It Works?
During training, check the printed logs for:

 Training loss
 
 Validation loss
 
 Validation accuracy
 
For predictions, add print() statements in predict.py to debug output probabilities and class labels.

Contributing
Want to improve this project? Feel free to submit a pull request!

