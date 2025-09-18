The model is a Multi-Layer Perceptron (MLP) designed to work with the 59 summary features from [[01 - Feature Engineering]]. It uses several techniques to improve performance and prevent overfitting.

### Key Components

- **Input Layer:** `Linear(in_features=59, out_features=512)`
- **Hidden Layers:** A series of dense layers with decreasing size (512 -> 256 -> 128 -> 64).
- **Activation Function:** `LeakyReLU` is used instead of standard ReLU to prevent "dying ReLU" issues.
- **Regularization:**
    - **`Dropout`**: Applied after each hidden layer to prevent co-adaptation of neurons.
    - **`BatchNorm1d`**: Used to stabilize training and speed up convergence.
- **Residual Blocks:** Skip connections are used to help with gradient flow in the deeper parts of the network, preventing vanishing gradients and allowing for a more complex model.
- **Output Layer:** `Linear(in_features=64, out_features=10)` providing raw logits for the 10 genres. `CrossEntropyLoss` is used, which includes a Softmax activation.