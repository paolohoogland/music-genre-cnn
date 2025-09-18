# Deployment: Frontend User Interface

The user interface is a single, self-contained HTML file located at `static/index.html`. It uses plain HTML, CSS, and JavaScript to interact with the [[04 - FastAPI Endpoint]].

### Key Features

1.  **File Upload:** A styled button that allows users to select a `.wav` file from their local machine.
2.  **Live Recording:** A second button that requests microphone access and records a 15-second `.wav` file directly in the browser using the `MediaRecorder` API.
3.  **Dynamic Feedback:** A status area provides real-time updates to the user (e.g., "Requesting microphone access...", "Recording... 10s left", "Classifying...", "Error: ...").
4.  **Results Display:**
    -   Shows the top predicted genre with its confidence score.
    -   Includes a collapsible "details" section that shows the confidence scores for all 10 genres, sorted from highest to lowest.
5.  **File Saving:** After a live recording is complete, it automatically triggers a download of the recorded `.wav` file for the user.

### How it Works (JavaScript Logic)

The core logic is handled by three main asynchronous functions:

- **`handleFileSelect()`:** Triggered when a user uploads a file. It gets the file and passes it to `sendAudioToServer()`.
- **`startRecording()`:** Triggered by the record button. It handles the microphone permission request, sets up the `MediaRecorder`, and starts the 30-second countdown. When the recording stops, it bundles the audio into a `.wav` file and passes it to `sendAudioToServer()`.
- **`sendAudioToServer(file)`:** This is the central communication function.
    1.  It takes a file object (either from upload or recording).
    2.  It creates a `FormData` object.
    3.  It uses the `fetch` API to send a `POST` request to the `http://127.0.0.1:8000/classify/` endpoint.
    4.  It handles the response, calling `displayResults()` on success or showing an error message on failure.

- **`displayResults(data)`:** This function takes the JSON response from the API and dynamically updates the HTML to show the main prediction and populate the details dropdown.