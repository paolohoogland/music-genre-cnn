# Code Structure

- **`Main.py`**: The main entry point. Handles argument parsing, orchestrates training, testing, and calls the classification logic.
- **`DatasetManager.py`**: A core class responsible for data preparation:
    - Extracts features from raw audio.
    - Saves and loads the feature CSV.
    - Splits data into train/val/test sets.
    - Applies feature scaling and saves the scaler.
    - Prepares single files for analysis.
- **`Audio.py`**: A helper class containing all the `librosa`-based feature extraction functions.
- **`MLP.py`**: Defines the [[02 - MLP Model Architecture]].
- **`api.py`**: The FastAPI application for serving the model.
- **`static/index.html`**: The frontend user interface.