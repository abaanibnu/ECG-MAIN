import os
import cv2
import numpy as np
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
from math import radians, cos, sin, asin, sqrt
from utils.image_processing import process_ecg_image
from utils.signal_analysis import analyze_ecg_signal
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({'success': True, 'filename': filename})
            
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/process')
def process_page():
    return render_template('process.html')

def crop_lead_region(image_path):
    """
    Extracts a clean Lead II region (Row 2, Column 1) from a 12-lead ECG image.
    Handles both 3-row and 4-row layouts.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image for cropping")

    height, width = img.shape[:2]

    # Detect row count: standard 12-lead is 3 rows or 4 rows (3 + rhythm strip)
    # If aspect ratio (H/W) is very small, it's likely 3 rows.
    # In the user image, aspect is ~400/1000 = 0.4.
    if height / width < 0.5:
        num_rows = 3
    else:
        num_rows = 4

    row_height = height // num_rows
    col_width = width // 4

    # Crop Lead II (Row 2, Column 1)
    # We add a small offset to avoid the lead labels (like "II") at the top/left
    y_start = row_height + int(row_height * 0.1)
    y_end = (row_height * 2) - int(row_height * 0.1)
    x_start = int(col_width * 0.05)
    x_end = col_width

    cropped = img[y_start:y_end, x_start:x_end]
    return cropped

def clean_image(cropped_img):
    """
    Apply adaptive thresholding to reduce grid noise and sharpen the waveform.
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    clean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    # Convert back to BGR so save/imwrite works consistently
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

def predict_ecg(image_path):
    """
    Existing logic wrapped for consistency. 
    Processes image path and returns analysis results.
    """
    signal_data = process_ecg_image(image_path)
    analysis_results = analyze_ecg_signal(signal_data)
    return signal_data, analysis_results

@app.route('/api/process', methods=['POST'])
def process_file():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
        
    filename = secure_filename(data['filename'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        height, width = img.shape[:2]
        
        # Detect if it's a 12-lead report:
        # 12-lead reports are landscape (wide) images.
        # Aspect ratio > 1.5 covers most scanned/photographed reports.
        aspect_ratio = width / height
        is_multi_lead = aspect_ratio > 1.5
        
        cropped_base64 = None
        
        if is_multi_lead:
            # --- Multi-Lead Path ---
            # 12-lead reports show ~2.5 seconds per lead column (10 sec / 4 cols)
            # We crop 1 column, so the effective duration is ~2.5 sec.
            one_col_width = width // 4
            col_duration = 2.5  # seconds per column in a standard report
            fs = one_col_width / col_duration
            
            # Crop Lead II (Row 2, Col 1)
            cropped_img = crop_lead_region(filepath)
            if cropped_img is None or cropped_img.size == 0:
                return jsonify({'error': 'Could not crop Lead II region'}), 400
            
            # Clean the cropped image
            cleaned_img = clean_image(cropped_img)
            
            # Save for analysis
            temp_filename = f"processed_{filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            cv2.imwrite(temp_path, cleaned_img)
            
            # Run analysis with calibrated fs
            signal_data = process_ecg_image(temp_path)
            analysis_results = analyze_ecg_signal(signal_data, fs=fs)
            
            # Encode cropped preview
            _, buffer = cv2.imencode('.png', cleaned_img)
            cropped_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            # --- Single-Lead Path ---
            # Single-lead images typically show 10 seconds of data.
            single_duration = 10.0
            fs = width / single_duration
            
            signal_data = process_ecg_image(filepath)
            analysis_results = analyze_ecg_signal(signal_data, fs=fs)
        
        return jsonify({
            'success': True,
            'signal': signal_data,
            'analysis': analysis_results,
            'is_multi_lead': is_multi_lead,
            'cropped_preview': cropped_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    try:
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract data
        patient = data.get('patient', {})
        analysis = data.get('analysis', {}) or {}
        chart_image_data = data.get('chartImage')
        is_multi_lead = data.get('is_multi_lead', False)
        cropped_preview = data.get('cropped_preview')

        print(f"DEBUG: Generating PDF for {patient.get('name')}. Multi-lead: {is_multi_lead}")

        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        elements = []
        styles = getSampleStyleSheet()

        # Custom Styles
        header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading1'], textColor=colors.HexColor('#0284c7'), fontSize=24, spaceAfter=5)
        sub_header_style = ParagraphStyle('SubHeaderStyle', parent=styles['Normal'], textColor=colors.HexColor('#64748b'), fontSize=12, spaceAfter=20)
        section_title_style = ParagraphStyle('SectionTitle', parent=styles['Heading2'], textColor=colors.HexColor('#1e293b'), fontSize=14, spaceBefore=15, spaceAfter=10, borderPadding=5)
        result_style = ParagraphStyle('ResultStyle', parent=styles['Normal'], fontSize=11, leading=14)

        # 1. Header Section
        elements.append(Paragraph("ECG Scan Helper", header_style))
        elements.append(Paragraph("Automated ECG Analysis Digital Report", sub_header_style))
        elements.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        # 2. Patient Information
        elements.append(Paragraph("Patient Information", section_title_style))
        rep_type = '12-Lead Report' if is_multi_lead else 'Single Lead'
        patient_data = [
            ['Name:', patient.get('name', 'N/A'), 'Age:', str(patient.get('age', 'N/A'))],
            ['Gender:', patient.get('gender', 'N/A'), 'Report Type:', rep_type]
        ]
        patient_table = Table(patient_data, colWidths=[1*inch, 2*inch, 1*inch, 1.5*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('TEXTCOLOR', (0,0), (0,-1), colors.grey),
            ('TEXTCOLOR', (2,0), (2,-1), colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.2*inch))

        # 3. ECG Metrics Summary
        elements.append(Paragraph("Cardiac Metrics Summary", section_title_style))
        
        abnormality = analysis.get('abnormality', 'Normal')
        cond_color = colors.HexColor('#10b981') if abnormality == 'Normal' else colors.HexColor('#ef4444')

        metrics_data = [
            ['Metric', 'Measured Value', 'Classification / Status'],
            ['Heart Rate', f"{analysis.get('heart_rate', 0)} BPM", abnormality],
            ['Stress Level', analysis.get('stress_level', 'N/A'), '-'],
            ['Result Summary', abnormality, 'Requires Review' if abnormality != 'Normal' else 'Normal']
        ]
        metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#475569')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('TEXTCOLOR', (2,1), (2,1), cond_color),
            ('FONTNAME', (2,1), (2,1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (2,3), (2,3), cond_color),
            ('FONTNAME', (2,3), (2,3), 'Helvetica-Bold'),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))

        # 4. ECG Waveform Analysis
        elements.append(Paragraph("ECG Waveform Analysis", section_title_style))
        
        if chart_image_data and "," in chart_image_data:
            try:
                elements.append(Paragraph("Full Waveform Visualization:", styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
                _, encoded_main = chart_image_data.split(",", 1)
                img_data_main = base64.b64decode(encoded_main)
                img_io_main = io.BytesIO(img_data_main)
                rl_img_main = RLImage(img_io_main, width=6*inch, height=2.2*inch)
                elements.append(rl_img_main)
                elements.append(Spacer(1, 0.2*inch))
            except Exception as img_err:
                print(f"DEBUG: Failed to add main chart image: {img_err}")

        if is_multi_lead and cropped_preview:
            try:
                elements.append(Paragraph("Lead II Extraction Detail:", styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
                encoded_crop = cropped_preview.split(",", 1)[1] if "," in cropped_preview else cropped_preview
                img_data_crop = base64.b64decode(encoded_crop)
                img_io_crop = io.BytesIO(img_data_crop)
                rl_img_crop = RLImage(img_io_crop, width=6*inch, height=1.2*inch)
                elements.append(rl_img_crop)
                elements.append(Spacer(1, 0.3*inch))
            except Exception as crop_err:
                print(f"DEBUG: Failed to add cropped preview: {crop_err}")

        # 5. Interpretation
        elements.append(Paragraph("Clinical Interpretation & Recommendation", section_title_style))
        elements.append(Paragraph(analysis.get('recommendation', 'No recommendation available.'), result_style))
        elements.append(Spacer(1, 0.5*inch))

        # 6. Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)
        elements.append(Paragraph("<b>Disclaimer:</b> This is an automated preliminary digital analysis provided by ECG Scan Helper. Informational only. No replacement for professional medical consultation.", disclaimer_style))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        filename = f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/nearest-hospitals", methods=["POST"])
def nearest_hospitals():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    query = f"""
    [out:json];
    (
      node[amenity=hospital][name~"Heart|Cardiac|Cardiology",i](around:10000,{lat},{lon});
      node[amenity=hospital](around:5000,{lat},{lon});
    );
    out;
    """

    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.get(url, params={"data": query}, timeout=10)
        response.raise_for_status()
        data_json = response.json()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch hospitals: {str(e)}"}), 500

    hospitals = []
    for el in data_json.get("elements", [])[:5]:
        name = el.get("tags", {}).get("name", "Hospital")
        phone = el.get("tags", {}).get("phone", "Not Available")
        h_lat = el["lat"]
        h_lon = el["lon"]

        distance = round(
            6371 * 2 * asin(sqrt(
            sin(radians(h_lat - lat) / 2)**2 +
            cos(radians(lat)) * cos(radians(h_lat)) *
            sin(radians(h_lon - lon) / 2)**2
        )), 2)

        hospitals.append({
            "name": name,
            "phone": phone,
            "distance": distance
        })

    return jsonify({"hospitals": hospitals})

@app.route("/aqi", methods=["POST"])
def get_aqi():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    # API_KEY instructions: Replace with actual key if available.
    API_KEY = "YOUR_OPENWEATHER_API_KEY" # Placeholder
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        r = response.json()
        
        if "list" not in r or not r["list"]:
            return jsonify({"error": "AQI data not available"}), 404
            
        aqi = r["list"][0]["main"]["aqi"]
    except Exception as e:
        return jsonify({"error": f"Failed to fetch AQI: {str(e)}"}), 500

    messages = {
        1: "Good air quality. Safe for outdoor activities.",
        2: "Moderate air quality. Sensitive people should be careful.",
        3: "Unhealthy for sensitive groups.",
        4: "Unhealthy air. Avoid outdoor activity.",
        5: "Very hazardous air quality."
    }

    return jsonify({
        "aqi": aqi,
        "message": messages.get(aqi, "Unknown condition")
    })

if __name__ == '__main__':
    app.run(debug=True)
