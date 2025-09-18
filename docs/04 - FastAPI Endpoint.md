The trained model is served via a web API built with **FastAPI**. This allows any application (like our [[04 - Frontend UI]] or any other program) to get genre predictions over the internet.

The main logic is in `api.py`.

### Startup Process

When the server starts (`uvicorn api:app`), it loads three critical components into memory once to ensure fast response times:

1.  **The Trained Model (`best_complex_model.pth`):** The PyTorch model weights are loaded into the [[02 - MLP Model Architecture]].
2.  **The Scaler (`scaler.joblib`):** The `StandardScaler` that was fit on the training data is loaded. This is essential for preprocessing new audio files in exactly the same way.
3.  **Metadata:** The list of all genre names is loaded from the feature CSV to map the model's output index back to a human-readable genre name.

### API Endpoints

The API exposes two endpoints:

#### 1. `GET /`
- **Purpose:** Serves the main `index.html` file, providing the user interface.
- **Method:** `GET`
- **Response:** An HTML page.

#### 2. `POST /classify/`
- **Purpose:** The core of the API. It accepts an audio file and returns a genre prediction.
- **Method:** `POST`
- **Request Body:** `multipart/form-data` containing a single file field named `file`.
- **Logic:**
    1.  Validates that the uploaded file is a `.wav`.
    2.  Saves the uploaded file to a temporary location on the server.
    3.  Uses the `_extract_single_file_features` method from `DatasetManager` to get the 59 summary features.
    4.  Uses the pre-loaded **scaler** to transform these features.
    5.  Converts the scaled features into a PyTorch tensor.
    6.  Feeds the tensor into the pre-loaded **model** to get the output logits.
    7.  Applies a `softmax` function to the logits to get confidence probabilities.
    8.  Finds the genre with the highest probability.
- **Response Body (JSON):**
```json
{
  "predicted_genre": "ROCK",
  "confidence_scores": {
    "blues": 5.32,
    "classical": 0.12,
    "rock": 35.00,
    "...": "..."
  }
}