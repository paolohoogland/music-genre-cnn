This section provides a complete guide on how to set up, train, and run the project from scratch. 
### Step 1: Set Up the Conda Environment

We use a Conda environment to ensure all dependencies are handled correctly.

1. **Create a new Conda environment** (this example uses Python 3.11, but 3.9+ should work).    
    ```
    conda create -n music-genre-env python=3.11
    ```
    
2. **Activate the newly created environment.** You must do this every time you work on the project.
    ```
    conda activate music-genre-env
    ```
    
3. **Install all required Python packages** using pip.
    ```
    pip install torch pandas numpy scikit-learn seaborn matplotlib librosa joblib
    pip install fastapi "uvicorn[standard]" python-multipart audiomentations
    ```

### Step 2: Download and Place the Dataset

The model is trained on the **GTZAN Music Genre Dataset**.

1. **Download the dataset from Kaggle:**
    
    - Go to the [GTZAN Dataset page on Kaggle](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fandradaolteanu%2Fgtzan-dataset-music-genre-classification).
        
    - Click "Download" and unzip the file.
        
2. **Place the data in the correct folder:**
    
    - Inside the project directory, create a folder named Data.
        
    - Find the genres_original folder from the unzipped download and move it inside the Data folder.
        
    
    Your final folder structure must look like this for the scripts to work:
    ```
    music-genre-cnn/
    ├── Data/
    │   └── genres_original/
    │       ├── blues/
    │       ├── classical/
    │       └── ... (10 genre folders in total)
    ├── Main.py
    └── ... (other project files)
    ```
### Step 3: Run the Full Data and Model Pipeline

The following two commands are **one-time setup steps** that will process the raw audio, create the feature dataset, and train the model.

1. **Create the Feature Dataset:**  
    This script will read all 1000 .wav files, extract features from each, and save the results into a single CSV file. This process will take several minutes.
    ```
    python Main.py --feature_dataset_creation
    ```
    
    This command will generate the Data/own_features_30_sec.csv file.
    
2. **Train the Model:**  
    Now, run the training script. This will load the feature CSV, split the data, train the MLP model, and save the two essential artifacts: the model weights and the data scaler.
    ```
    python Main.py
    ```
    
    This command will generate best_complex_model.pth and scaler.joblib in your project's root directory.
    

**Setup is now complete!** You can now run the web application.

### Step 5: Run the Web Application

To use the interactive web interface, start the FastAPI server.

1. **Start the Server:**
    ```
    uvicorn api:app --reload
    ```
    
2. **Open Your Browser:**  
    Navigate to **[http://127.0.0.1:8000](https://www.google.com/url?sa=E&q=http%3A%2F%2F127.0.0.1%3A8000)**.
    

You can now upload a .wav file or record live audio to classify its genre.