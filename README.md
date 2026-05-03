# fashion-mnist-neural-network-classification
Neural network classification and probabilistic visualization using Fashion-MNIST with animated prediction interpretation.

Post: https://www.linkedin.com/posts/brenogdsilva_nem-toda-evolu%C3%A7%C3%A3o-em-ia-come%C3%A7a-com-um-modelo-activity-7452740512911941632-7nl6?utm_source=share&utm_medium=member_desktop&rcm=ACoAADzYEn0BidO853LZcLlVhNbNq-HOPuFNURk

# Neural Network Classification with Fashion-MNIST: A Visual Approach

## Introduction

Image classification is a central problem in machine learning and computer vision, where the goal is to assign a label to an input image based on learned patterns. With the rise of deep learning, neural networks have become the dominant approach for solving such tasks (LeCun et al., 2015).

However, despite their predictive power, neural networks are often considered "black-box" models, making it difficult to interpret how predictions are formed. From a data science perspective, understanding the relationship between input data, learned representations, and output probabilities is essential.

This project provides a visual and statistical interpretation of a neural network trained on the Fashion-MNIST dataset, illustrating how raw pixel data is transformed into probabilistic predictions.

## Dataset

The Fashion-MNIST dataset (Xiao et al., 2017) consists of grayscale images of size $28 \times 28$, representing 10 categories of clothing items.

Each observation is defined as:

$$
\mathbf{x}_i \in \mathbb{R}^{28 \times 28}, \quad y_i \in \{0,1,\dots,9\}
$$

where:

- $\mathbf{x}_i$ is the image
- $y_i$ is the class label

The dataset is split into training and testing sets.

## Methodology

### Neural Network Model

The model used is a feedforward neural network defined as:

$$
f(\mathbf{x}) = W^{(2)} \cdot \sigma \left( W^{(1)} \cdot \text{vec}(\mathbf{x}) + b^{(1)} \right) + b^{(2)}
$$

where:

- $\text{vec}(\mathbf{x})$ flattens the image
- $\sigma(\cdot)$ is the ReLU activation function
- $W^{(1)}, W^{(2)}$ are weight matrices
- $b^{(1)}, b^{(2)}$ are bias terms

The hidden layer uses:

$$
\sigma(z) = \max(0, z)
$$

### Loss Function

The model is trained using the sparse categorical cross-entropy:

$$
\mathcal{L} = - \sum_{i=1}^{N} \log p_{y_i}(\mathbf{x}_i)
$$

where $p_{y_i}$ is the predicted probability for the true class.

### Softmax Function

The final layer outputs logits, which are converted into probabilities using the softmax function:

$$
p_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}
$$

ensuring:

$$
\sum_{k=1}^{K} p_k = 1
$$

### Optimization

The model parameters are optimized using the Adam algorithm (Kingma & Ba, 2014), which combines adaptive learning rates and momentum.

## Visualization Approach

The project introduces an animated visualization to illustrate the classification process in three stages:

1. Pixel intensity representation (heatmap)
2. Original grayscale image
3. Predicted class probabilities

This provides an intuitive mapping between:

$$
\mathbf{x} \rightarrow f(\mathbf{x}) \rightarrow \mathbf{p}
$$

where $\mathbf{p}$ is the probability vector.

## Results

The model successfully learns discriminative patterns across clothing categories.

For each image, the output is a probability vector:

$$
\mathbf{p} = (p_1, p_2, \dots, p_{10})
$$

The predicted class is given by:

$$
\hat{y} = \arg\max_k p_k
$$

The animation highlights:

- Confidence levels of predictions
- Misclassification patterns
- Distribution of probabilities across classes

From a statistical perspective, the model estimates the conditional probability:

$$
P(Y = k \mid X = \mathbf{x})
$$

which is the fundamental objective of classification models.

## Conclusion

This project demonstrates how neural networks can be interpreted through visualization and probabilistic outputs.

Beyond predictive accuracy, understanding how models transform data into probabilities is crucial for:

- model interpretability
- decision-making
- trust in AI systems

The animation provides an intuitive bridge between raw data and statistical inference in deep learning.

## References

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset.
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
