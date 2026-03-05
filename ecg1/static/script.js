
document.addEventListener('DOMContentLoaded', () => {
    // --- Upload Page Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const validationMsg = document.getElementById('validation-msg');
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const btnRemove = document.getElementById('btn-remove');
    const btnProceed = document.getElementById('btn-proceed');

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    const districtSelect = document.getElementById('district-select');
    const coordsDisplay = document.getElementById('selected-coords-display');

    // Force fresh location selection on every page load
    if (dropZone) {
        localStorage.removeItem('user_district');
        if (districtSelect) districtSelect.value = "";
        if (btnProceed) btnProceed.disabled = true;
    }

    let selectedFile = null;

    // --- Dummy Kerala Data Mapping ---
    const keralaData = {
        "Alappuzha": {
            aqi: 38,
            message: "Good: Air quality is satisfactory. Enjoy the backwaters!",
            hospitals: [
                { name: "Sahrudaya Cardiac Centre", phone: "0478 281 3131" },
                { name: "TD Medical College (Cardiology)", phone: "0477 228 2015" }
            ]
        },
        "Ernakulam": {
            aqi: 72,
            message: "Moderate: Sensitive individuals should limit outdoor exertion.",
            hospitals: [
                { name: "Amrita Institute of Medical Sciences (Cardiology)", phone: "0484 280 1234" },
                { name: "Lourdes Heart Institute", phone: "0484 412 3456" },
                { name: "Aster Medcity Cardiac Sciences", phone: "0484 669 9999" }
            ]
        },
        "Idukki": {
            aqi: 25,
            message: "Excellent: Perfect time for high-altitude walks.",
            hospitals: [
                { name: "St. John's Hospital (Cardiac Care)", phone: "04863 222 222" },
                { name: "Idukki District Hospital Cardiology Unit", phone: "04862 232 245" }
            ]
        },
        "Kannur": {
            aqi: 45,
            message: "Good: Enjoy the coastal breeze.",
            hospitals: [
                { name: "AKG Memorial Cooperative Hospital (Cardiology)", phone: "0497 270 1311" },
                { name: "Kannur Medical College (Cardiac Dept)", phone: "0497 285 5000" }
            ]
        },
        "Kasaragod": {
            aqi: 32,
            message: "Good: Air is clean and fresh.",
            hospitals: [
                { name: "Kasaragod Institute of Medical Sciences", phone: "04994 220 000" },
                { name: "Malabar Cardiac Centre (Unit II)", phone: "04994 231 456" }
            ]
        },
        "Kollam": {
            aqi: 54,
            message: "Moderate: Acceptable air quality.",
            hospitals: [
                { name: "N.S. Memorial Institute of Medical Sciences (Cardiology)", phone: "0474 272 3191" },
                { name: "Bishop Benziger Hospital (Heart Centre)", phone: "0474 274 8181" }
            ]
        },
        "Kottayam": {
            aqi: 48,
            message: "Good: Great time to visit the hills.",
            hospitals: [
                { name: "Caritas Hospital Heart Institute", phone: "0481 279 0025" },
                { name: "Kottayam Medical College (Cardiology Dept)", phone: "0481 257 3251" },
                { name: "MGM Muthoot Heart Institute", phone: "0481 256 0864" }
            ]
        },
        "Kozhikode": {
            aqi: 65,
            message: "Moderate: Mostly fine, sensitive groups take care.",
            hospitals: [
                { name: "Baby Memorial Hospital (Cardiac Centre)", phone: "0495 272 3272" },
                { name: "MIMS Cardiac Sciences", phone: "0495 248 8000" },
                { name: "Calicut Medical College Super Speciality (Cardio)", phone: "0495 235 0212" }
            ]
        },
        "Malappuram": {
            aqi: 52,
            message: "Moderate: Normal outdoors today.",
            hospitals: [
                { name: "Moulana Hospital (Cardiology Unit)", phone: "0483 273 0000" },
                { name: "MES Medical College Cardiology", phone: "04933 298 300" }
            ]
        },
        "Palakkad": {
            aqi: 58,
            message: "Moderate: Slightly dry air today.",
            hospitals: [
                { name: "Lakshmi Hospital (Cardiac Care)", phone: "0491 253 9130" },
                { name: "Ahalia Heart Foundation", phone: "04923 235 888" }
            ]
        },
        "Pathanamthitta": {
            aqi: 35,
            message: "Good: Healthy air for outdoor activities.",
            hospitals: [
                { name: "Muthoot Heart Centre", phone: "0468 222 2222" },
                { name: "Pushpagiri Heart Institute", phone: "0469 270 0755" }
            ]
        },
        "Thiruvananthapuram": {
            aqi: 68,
            message: "Moderate: Urban air levels, generally okay.",
            hospitals: [
                { name: "Sree Chitra Tirunal Institute (SCTIMST)", phone: "0471 244 3152" },
                { name: "PRS Hospital Cardiac Centre", phone: "0471 234 4443" },
                { name: "KIMS Cancer & Heart Institute", phone: "0471 304 1000" }
            ]
        },
        "Thrissur": {
            aqi: 62,
            message: "Moderate: Take standard precautions.",
            hospitals: [
                { name: "Jubilee Mission Medical College (Cardiology)", phone: "0487 243 2200" },
                { name: "West Fort Hi-Tech Heart Institute", phone: "0487 238 2382" }
            ]
        },
        "Wayanad": {
            aqi: 18,
            message: "Excellent: Cleanest air in the state. Breath deep!",
            hospitals: [
                { name: "Wayanad District Hospital (Cardiac Clinic)", phone: "04936 220 252" },
                { name: "DM WIMS Heart Centre", phone: "04936 287 000" }
            ]
        }
    };

    if (dropZone) {
        // Handle District selection
        if (districtSelect) {
            districtSelect.addEventListener('change', (e) => {
                const district = e.target.value;
                if (district && keralaData[district]) {
                    localStorage.setItem('user_district', district);
                    if (coordsDisplay) {
                        coordsDisplay.textContent = `✅ Selected: ${district}`;
                        coordsDisplay.classList.remove('hidden');
                    }
                    if (selectedFile) btnProceed.disabled = false;
                } else {
                    localStorage.removeItem('user_district');
                    if (coordsDisplay) coordsDisplay.classList.add('hidden');
                    btnProceed.disabled = true;
                }
            });
        }

        // Drag & Drop Handlers
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFileSelection(e.dataTransfer.files[0]);
        });

        dropZone.addEventListener('click', (e) => { if (e.target.tagName !== 'BUTTON') fileInput.click(); });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFileSelection(e.target.files[0]); });

        btnRemove.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            previewSection.classList.add('hidden');
            dropZone.classList.remove('hidden');
            hideError();
            btnProceed.disabled = true;
        });

        btnProceed.addEventListener('click', () => {
            if (!selectedFile || !localStorage.getItem('user_district')) return;

            btnProceed.disabled = true;
            btnRemove.classList.add('hidden');
            progressContainer.classList.remove('hidden');
            loadingSpinner.classList.remove('hidden');
            hideError();

            const formData = new FormData();
            formData.append('file', selectedFile);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = Math.floor((e.loaded / e.total) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressText.textContent = percentComplete + '%';
                }
            };

            xhr.onload = function () {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        localStorage.setItem('ecg_filename', response.filename || selectedFile.name);
                        setTimeout(() => { window.location.href = '/process'; }, 500);
                    } else {
                        showError(response.error || 'Upload failed');
                        resetUploadUI();
                    }
                } else {
                    showError('Server error during upload.');
                    resetUploadUI();
                }
            };
            xhr.send(formData);
        });
    }

    function handleFileSelection(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) { showError('Invalid file type.'); return; }
        hideError();
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewSection.classList.remove('hidden');

            // Strictly enforce location selection
            if (localStorage.getItem('user_district')) {
                btnProceed.disabled = false;
            } else {
                btnProceed.disabled = true;
            }
        };
        reader.readAsDataURL(file);
    }

    function showError(msg) { if (validationMsg) { validationMsg.textContent = msg; validationMsg.classList.remove('hidden'); } }
    function hideError() { if (validationMsg) validationMsg.classList.add('hidden'); }
    function resetUploadUI() {
        if (localStorage.getItem('user_district') && selectedFile) {
            btnProceed.disabled = false;
        } else {
            btnProceed.disabled = true;
        }
        btnRemove.classList.remove('hidden');
        progressContainer.classList.add('hidden');
        loadingSpinner.classList.add('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
    }

    // --- Dashboard Logic ---
    const dashboardContainer = document.getElementById('dashboard-container');
    if (dashboardContainer) {
        const loader = document.getElementById('loader-container');
        const alertPanel = document.getElementById('alert-panel');
        const alertMessage = document.getElementById('alert-message');
        const metricHr = document.getElementById('val-hr');
        const metricStress = document.getElementById('val-stress');
        const metricCondition = document.getElementById('val-condition');
        const recommendationText = document.getElementById('val-recommendation');
        const btnResetZoom = document.getElementById('reset-zoom');
        const hospitalList = document.getElementById('hospital-list');
        const aqiValue = document.getElementById('aqi-value');
        const aqiMessage = document.getElementById('aqi-message');
        const intelligentFeatures = document.getElementById('intelligent-features');

        let ecgChart = null;
        const filename = localStorage.getItem('ecg_filename');

        if (!filename) { window.location.href = '/'; return; }

        let latestAnalysis = null;

        fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename })
        })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    latestAnalysis = data; // Store the full response including cropped_preview and is_multi_lead
                    renderDashboard(data);
                } else {
                    alert('Analysis failed: ' + data.error);
                    window.location.href = '/';
                }
            });

        function renderDashboard(data) {
            const analysis = data.analysis;
            loader.classList.add('hidden');
            dashboardContainer.classList.remove('hidden');

            metricHr.textContent = analysis.heart_rate;
            metricStress.textContent = analysis.stress_level;
            metricCondition.textContent = analysis.abnormality;
            recommendationText.textContent = analysis.recommendation;


            if (analysis.abnormality !== 'Normal') {
                alertPanel.classList.remove('hidden');
                alertMessage.textContent = analysis.recommendation;

                const district = localStorage.getItem('user_district');
                if (district && keralaData[district]) {
                    displayDummyFeatures(district);
                }
            }
            renderChart(data.signal);
        }

        function renderChart(signalData) {
            const ctx = document.getElementById('ecgChart').getContext('2d');
            const labels = Array.from({ length: signalData.length }, (_, i) => (i * 10).toFixed(0));
            ecgChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'ECG Signal (mV)',
                        data: signalData,
                        borderColor: '#0ea5e9',
                        backgroundColor: 'rgba(14, 165, 233, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { display: false }, y: { beginAtZero: false } }
                }
            });
            if (btnResetZoom) btnResetZoom.addEventListener('click', () => ecgChart.resetZoom());
        }

        function displayDummyFeatures(district) {
            if (intelligentFeatures) intelligentFeatures.classList.remove('hidden');
            const data = keralaData[district];

            if (aqiValue) aqiValue.textContent = data.aqi;
            if (aqiMessage) aqiMessage.textContent = data.message;

            if (hospitalList) {
                let html = '';
                data.hospitals.forEach(h => {
                    html += `
                    <div class="metric-item" style="border-bottom: 1px solid #fee2e2; padding: 0.8rem 0;">
                        <div class="metric-info">
                            <span class="metric-label" style="color: #9f1239; font-weight: 600; font-size: 1rem;">${h.name}</span>
                            <div class="metric-value-container">
                                <span class="metric-unit" style="font-size: 0.85rem; color: #e11d48; background: #fff1f2; padding: 2px 8px; border-radius: 4px; border: 1px solid #fecdd3;">📞 ${h.phone}</span>
                            </div>
                        </div>
                    </div>`;
                });
                hospitalList.innerHTML = html;
            }
        }

        // PDF Generation
        const btnDownloadPdf = document.getElementById('btn-download-pdf');
        if (btnDownloadPdf) {
            btnDownloadPdf.addEventListener('click', () => {
                if (!latestAnalysis) return;
                const chartImage = document.getElementById('ecgChart').toDataURL('image/png');
                const patientData = {
                    name: document.getElementById('patient-name').value || 'N/A',
                    age: document.getElementById('patient-age').value || 'N/A',
                    gender: document.getElementById('patient-gender').value || 'N/A'
                };
                fetch('/generate-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        patient: patientData,
                        analysis: latestAnalysis.analysis, // Pass the inner analysis object
                        is_multi_lead: latestAnalysis.is_multi_lead,
                        cropped_preview: latestAnalysis.cropped_preview,
                        chartImage: chartImage
                    })
                })
                    .then(async res => {
                        if (!res.ok) {
                            const errorData = await res.json().catch(() => ({}));
                            throw new Error(errorData.error || 'PDF Generation failed');
                        }
                        return res.blob();
                    })
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `ECG_Report_${new Date().getTime()}.pdf`;
                        a.click();
                    })
                    .catch(err => {
                        console.error(err);
                        alert('Error generating PDF: ' + err.message);
                    });
            });
        }
    }
});
