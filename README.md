# Handwritten-Digit-Recognition-Using-Neural-Networks
A neural network model for recognizing handwritten digits from the MNIST dataset with 97% accuracy. Includes a user-friendly GUI for real-time digit predictions. Built with Python, NumPy, and Tkinter, showcasing core machine learning and neural network techniques.

# Table of Contents
- Introduction
- Dataset Link
- Methodology
- Installation
- How to run the Digit Recognition Model
- Conclusion
- References

# Introduction
Using the MNIST dataset, handwritten digit recognition is a significant effort that was created with the use of neural networks. In essence, it recognizes the scanned copies of handwritten numbers. Our handwritten digit recognition technology goes a step further by allowing users to write their own numbers on the screen using an integrated GUI for recognition in addition to detecting handwritten digits in scanned photos.
The brightness of each pixel on a greyscale of CNN's 28x28 grid, which has 784 pixels, represents the neuron's "activation," with 0 being black and 1 denoting white.The initial layer of the neural network is made up of each of these 784 neurons. Ten neurons representing one digit each make up the last layer. Based on which neuron in the network has the highest level of activation, the final layer identifies which digit a picture represents.
The goal is for the neural network to move from pixels to edges, edges to patterns, and patterns to digits.

# Dataset Link
[Dataset Link](https://www.kaggle.com/datasets/avnishnish/mnist-original?resource=download]).

# Methodology
This project focuses on building a neural network model to recognize handwritten digits using the MNIST dataset, leveraging Python and libraries like SciPy and NumPy. Here’s an outline of the approach:

1. **Dataset**: The MNIST dataset of handwritten digits (0-9) is used, comprising 60,000 training and 10,000 testing images. Each image is a 28x28 pixel grayscale image, flattened to a 784-dimensional input vector.
   
2. **Model Architecture**:<br>
i) **Input Layer**: 784 neurons (one for each pixel in the 28x28 image).<br>
ii) **Hidden Layer**: 100 neurons with a sigmoid activation function.<br>
iii) **Output Layer**: 10 neurons (one for each digit 0-9), with softmax for multi-class classification.<br>

3. **Data Processing**:
The images are normalized by scaling pixel values to a [0, 1] range to prevent overflow during computation.
The data is split into training (60,000 examples) and testing sets (10,000 examples).

4. **Training Process**:<br>
a) **Initialization**: Model weights (Theta1 and Theta2) are initialized randomly with small values to ensure uniform learning.<br>
b) **Feedforward Propagation**: The data is passed through the network, and activations are computed layer-by-layer. The sigmoid function is used as the activation function in the hidden layer, producing probabilistic predictions at the output layer.<br>
c) **Backpropagation**: Gradients of the cost function (cross-entropy) are calculated with respect to each weight, using L-BFGS-B optimization to minimize the cost function and update weights.<br>
d) **Regularization**: A regularization parameter (lambda = 0.1) is used to address overfitting by penalizing high weights.<br>
e) **Optimizer**: The L-BFGS-B algorithm is used with a maximum of 70 iterations to optimize the weights, aiming to reach the optimal solution efficiently.<br>

5. **Evaluation**:
The model's performance is evaluated on the test set by measuring accuracy and precision. Typical results achieve 97.32% test accuracy and over 99.4% training accuracy.

6. **Deployment**:
The trained weights are saved in Theta1.txt and Theta2.txt, allowing for easy model reloading and further testing.
A graphical user interface (GUI) is implemented to allow users to draw digits on the screen and have them recognized by the trained model.

# Installation Guide

Installation Guide
Follow these steps to set up the project on your local machine.

1. **Install Python**:<br>
Ensure that Python 3.x is installed on your machine. You can download Python from [python.org](https://www.python.org/downloads/).

Verify the installation:<br>
```bash
python --version
```

2. **Clone the Repository**:<br>
First, clone this repository to your local machine.
```bash
git clone [repository-link]
```

3. **Navigate into the project directory**:<br>
```bash
cd your-repo-name
```

4. **Create a Virtual Environment (Optional but Recommended)**:<br>
It’s best to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate        # For macOS/Linux
.\venv\Scripts\activate         # For Windows
```

5. **Install Project Dependencies**:

Use the pip package manager to install the required libraries:
```bash
pip install -r requirements.txt
```
Note: If requirements.txt does not exist, you can manually install dependencies:
```bash
pip install numpy scipy pillow tkinter
```

6. **Download the MNIST Dataset**:

Download the MNIST dataset file mentioned in the dataset link section.
Place the downloaded file in the root directory of the project, or specify its path in the code if stored elsewhere.

7. Train the Model (Optional)
To train the model and generate the weight files Theta1.txt and Theta2.txt:
```bash
python Main.py
```
This script will load the dataset, train the model, and save the trained weights. Training results will display in the console.

7. ***Run the GUI for Digit Prediction**:

To test the model with a GUI, use the following command:
```bash
python GUI.py
```
This will open a graphical interface where you can draw digits to be recognized by the model.

# How to run the Digit Recognition Model

Examples
This section provides examples of how to use the digit recognition model, both through code and the GUI interface.

1. **Using the Model with Code**:
You can make predictions on images programmatically using the pre-trained model. Here’s an example of how to do this:<br>

1. Load the Model Weights: Load the trained weights saved in Theta1.txt and Theta2.txt.<br>
2. Prepare an Image: The input image should be a 28x28 grayscale image, flattened into a (1, 784) numpy array and normalized to [0, 1].<br>
3. Make a Prediction:
```bash
import numpy as np

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    
    # Add bias unit to input layer
    X = np.append(one_matrix, X, axis=1)
    
    # Hidden layer activations
    z2 = np.dot(X, Theta1.T)
    a2 = 1 / (1 + np.exp(-z2))
    a2 = np.append(np.ones((a2.shape[0], 1)), a2, axis=1)  # Add bias to hidden layer
    
    # Output layer activations
    z3 = np.dot(a2, Theta2.T)
    a3 = 1 / (1 + np.exp(-z3))
    
    # Prediction: index of the max activation in output layer
    p = np.argmax(a3, axis=1)
    
    return p
```
This code demonstrates the prediction phase of a neural network model, specifically for handwritten digit recognition. It uses trained weights (Theta1 and Theta2) to make predictions based on input data (X).

2. **Using the GUI for Digit Recognition**:<br>
The project includes a graphical interface where users can draw digits to be recognized by the trained model.

i). **Run the GUI Script**:
```bash
python GUI.py
```
ii) **Draw a Digit**: Use your mouse to draw a digit (0-9) on the canvas in the GUI window.<br>
iii) **Predict the Digit**:<br>
     - Click the Predict button to generate a prediction.<br>
     - The recognized digit will display on the GUI.<br>

3. **Evaluating the Model on the MNIST Test Set**:<br>
To check the model’s performance, you can evaluate it on the MNIST test dataset.

1. **Run Main.py**:
```bash
python Main.py
```
2. **Output**: The script will output the model’s accuracy on the test set and training set, showing how well it recognizes digits.

# Conclusion

This project demonstrates how neural networks can be applied to digit recognition. The high accuracy achieved indicates the robustness of this approach for image classification tasks.

# References<br>

-Ian Goodfellow, Yoshua Bengio, Aaron Courville, “Deep Learning (Adaptive Computation and Machine Learning series)”, The MIT Press, 2016.<br>
-V. Susheela Devi, M. Narasimha Murty, “Pattern Recognition: An Introduction”, University Press,
