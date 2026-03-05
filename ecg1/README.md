# ECG Scan Helper

A web-based tool to convert ECG paper images into digital signals and analyze them for heart anomalies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    python app.py
    ```

3.  **Access**:
    Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Features
-   **Image Digitization**: Converts scanned ECGs to digital waveforms.
-   **Signal Analysis**: Detects R-peaks and calculates Heart Rate.
-   **Health Insights**: Identifies Bradycardia/Tachycardia and estimates stress levels.

## Tech Stack
-   **Backend**: Flask (Python)
-   **Image Processing**: OpenCV, NumPy
-   **Signal Processing**: SciPy
-   **Frontend**: HTML5, CSS3, JavaScript, Chart.js
