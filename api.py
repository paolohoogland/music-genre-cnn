import os
import joblib
import torch
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from MLP import MLP
from DatasetManager import DatasetManager

MODEL_PATH = "best_complex_model.pth"
SCALER_PATH = "scaler.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load helper class to get genre list and feature names
try:
    manager = DatasetManager()
    manager.get_feature_dataset()
    ALL_GENRES = manager.all_genres_list
    INDEX_TO_GENRE = {i: genre for i, genre in enumerate(ALL_GENRES)}
    NUM_GENRES = len(ALL_GENRES)
except Exception as e:
    print(f"Warning: Could not load initial manager data: {e}")
    ALL_GENRES, INDEX_TO_GENRE, NUM_GENRES = [], {}, 0

# The scaler is responsible for feature scaling, which means it standardizes the input features to have a mean of 0 and a variance of 1
try:
    SCALER = joblib.load(SCALER_PATH)
    SCALER_FEATURES = SCALER.get_feature_names_out()
    NUM_FEATURES = len(SCALER_FEATURES)
    print(f"Scaler loaded successfully. Expecting {NUM_FEATURES} features.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Scaler file not found at '{SCALER_PATH}'. API cannot function.")
    SCALER, NUM_FEATURES = None, 0

# Load the trained model
if NUM_GENRES > 0 and NUM_FEATURES > 0:
    MODEL = MLP(num_genres=NUM_GENRES, num_features=NUM_FEATURES)
    try:
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        print("PyTorch model loaded successfully.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at '{MODEL_PATH}'. API cannot function.")
        MODEL = None
    except Exception as e:
        print(f"CRITICAL ERROR: Error loading model: {e}")
        MODEL = None
else:
    MODEL = None

app = FastAPI(title="Music Genre Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class ClassificationResponse(BaseModel):
    predicted_genre: str
    confidence_scores: dict[str, float]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=500)

@app.post("/classify/", response_model=ClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded. API is not ready.")
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
        content = await file.read()
        temp_audio_file.write(content)
        
        manager = DatasetManager()
        raw_features_df = manager._extract_single_file_features(temp_audio_file.name)

        if raw_features_df is None:
            raise HTTPException(status_code=500, detail="Failed to extract features from the audio file.")
        
        try:
            raw_features_df = raw_features_df[SCALER_FEATURES]
        except KeyError:
            raise HTTPException(status_code=500, detail="Feature mismatch between audio and model.")
        
        scaled_features = SCALER.transform(raw_features_df)
        
    with torch.no_grad():
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
        output = MODEL(features_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_genre = INDEX_TO_GENRE[predicted_index]

    confidence_scores = {genre: probabilities[0, i].item() * 100 for i, genre in INDEX_TO_GENRE.items()}

    return {
        "predicted_genre": predicted_genre.upper(),
        "confidence_scores": confidence_scores
    }