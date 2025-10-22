// MNIST Digit Recognition JavaScript

// Canvas drawing variables
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 15;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeDrawingCanvas();
    loadModelInfo();
    setupEventListeners();
});

function setupEventListeners() {
    // Brush size slider
    const brushSlider = document.getElementById('brushSize');
    const brushSizeValue = document.getElementById('brushSizeValue');
    
    brushSlider.addEventListener('input', function() {
        brushSize = parseInt(this.value);
        brushSizeValue.textContent = brushSize;
    });

    // Image preview for file upload
    const imageFile = document.getElementById('imageFile');
    imageFile.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('imagePreview');
                preview.src = e.target.result;
                preview.style.display = 'block';
                document.getElementById('noPreview').style.display = 'none';
            }
            reader.readAsDataURL(file);
        }
    });
}

// Drawing canvas functions
function initializeDrawingCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas background to black (MNIST style)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing style
    ctx.strokeStyle = '#ffffff';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = brushSize;
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
}

function startDrawing(e) {
    isDrawing = true;
    const canvas = document.getElementById('drawingCanvas');
    const rect = canvas.getBoundingClientRect();
    [lastX, lastY] = getMousePos(canvas, e);
    updatePreview();
}

function draw(e) {
    if (!isDrawing) return;
    
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const [currentX, currentY] = getMousePos(canvas, e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    [lastX, lastY] = [currentX, currentY];
    updatePreview();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    
    if (e.type === 'touchstart') {
        startDrawing(mouseEvent);
    } else if (e.type === 'touchmove') {
        draw(mouseEvent);
    }
}

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    
    if (evt.type.includes('touch')) {
        clientX = evt.touches[0].clientX;
        clientY = evt.touches[0].clientY;
    } else {
        clientX = evt.clientX;
        clientY = evt.clientY;
    }
    
    return [
        clientX - rect.left,
        clientY - rect.top
    ];
}

function updatePreview() {
    const drawingCanvas = document.getElementById('drawingCanvas');
    const previewCanvas = document.getElementById('previewCanvas');
    const previewCtx = previewCanvas.getContext('2d');
    
    // Clear preview
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
    
    // Draw scaled version
    previewCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
}

function clearCanvas() {
    const drawingCanvas = document.getElementById('drawingCanvas');
    const previewCanvas = document.getElementById('previewCanvas');
    const ctx = drawingCanvas.getContext('2d');
    const previewCtx = previewCanvas.getContext('2d');
    
    // Clear both canvases
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
    
    // Hide results
    document.getElementById('drawingResults').style.display = 'none';
    document.getElementById('noDrawingResults').style.display = 'block';
}

// Prediction functions
async function predictDrawing() {
    const previewCanvas = document.getElementById('previewCanvas');
    
    // Convert canvas to blob
    previewCanvas.toBlob(async function(blob) {
        const formData = new FormData();
        formData.append('image', blob, 'digit.png');
        
        try {
            const response = await fetch('/task2/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayDrawingResults(result.prediction);
            } else {
                alert('Prediction error: ' + result.error);
            }
        } catch (error) {
            alert('Network error: ' + error.message);
        }
    }, 'image/png');
}

function displayDrawingResults(prediction) {
    document.getElementById('noDrawingResults').style.display = 'none';
    document.getElementById('drawingResults').style.display = 'block';
    
    document.getElementById('drawingPrediction').textContent = prediction.predicted_digit;
    document.getElementById('drawingConfidence').textContent = (prediction.confidence * 100).toFixed(2) + '%';
    
    // Display probabilities
    let probHtml = '';
    for (const [digit, prob] of Object.entries(prediction.probabilities)) {
        const percentage = (prob * 100).toFixed(2);
        const isPredicted = digit === prediction.predicted_digit.toString();
        
        probHtml += `
            <div class="mb-2">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="${isPredicted ? 'fw-bold text-primary' : ''}">${digit}</span>
                    <span class="${isPredicted ? 'fw-bold text-primary' : ''}">${percentage}%</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar ${isPredicted ? 'bg-primary' : 'bg-secondary'}" 
                         style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }
    document.getElementById('drawingProbabilities').innerHTML = probHtml;
}

// File upload prediction
document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Predicting...';
    submitBtn.disabled = true;
    
    try {
        const response = await fetch('/task2/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictionResults(result.prediction);
        } else {
            alert('Prediction error: ' + result.error);
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

function displayPredictionResults(prediction) {
    document.getElementById('noResults').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';
    
    document.getElementById('predictedDigit').textContent = prediction.predicted_digit;
    document.getElementById('confidenceValue').textContent = (prediction.confidence * 100).toFixed(2) + '%';
    
    // Display probabilities
    let probHtml = '';
    for (const [digit, prob] of Object.entries(prediction.probabilities)) {
        const percentage = (prob * 100).toFixed(2);
        const isPredicted = digit === prediction.predicted_digit.toString();
        
        probHtml += `
            <div class="mb-2">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="${isPredicted ? 'fw-bold text-success' : ''}">${digit}</span>
                    <span class="${isPredicted ? 'fw-bold text-success' : ''}">${percentage}%</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar ${isPredicted ? 'bg-success' : 'bg-secondary'}" 
                         style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }
    document.getElementById('probabilityChart').innerHTML = probHtml;
    
    // Display model info
    if (prediction.model_info) {
        const modelInfo = prediction.model_info;
        document.getElementById('modelInfo').innerHTML = `
            <p class="mb-1"><strong>Type:</strong> ${modelInfo.model_type}</p>
            <p class="mb-1"><strong>Input Shape:</strong> ${modelInfo.input_shape.join('x')}</p>
            <p class="mb-1"><strong>Classes:</strong> ${modelInfo.num_classes}</p>
            <p class="mb-0"><strong>Trained:</strong> ${modelInfo.training_date}</p>
        `;
    }
}

// Training functions
document.getElementById('trainForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    document.getElementById('trainingProgress').style.display = 'block';
    document.getElementById('trainingResults').style.display = 'none';
    
    const formData = new FormData(this);
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Training...';
    submitBtn.disabled = true;
    
    // Simulate progress updates
    const progressStages = [
        {message: "Loading MNIST dataset...", progress: 10},
        {message: "Preprocessing data...", progress: 25},
        {message: "Building CNN model...", progress: 40},
        {message: "Starting training...", progress: 55},
        {message: "Training in progress...", progress: 75},
        {message: "Evaluating model...", progress: 90},
        {message: "Generating visualizations...", progress: 100}
    ];
    
    let currentStage = 0;
    const progressInterval = setInterval(() => {
        if (currentStage < progressStages.length) {
            const stage = progressStages[currentStage];
            const progressBar = document.getElementById('progressBar');
            const progressMessages = document.getElementById('progressMessages');
            
            progressBar.style.width = stage.progress + '%';
            progressBar.textContent = stage.progress + '%';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'text-success small';
            messageDiv.innerHTML = `<i class="bi bi-check-circle"></i> ${stage.message}`;
            progressMessages.appendChild(messageDiv);
            
            currentStage++;
        }
    }, 1500);
    
    try {
        const response = await fetch('/task2/train', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        const result = await response.json();
        
        if (result.success) {
            displayTrainingResults(result);
        } else {
            alert('Training error: ' + result.error);
        }
    } catch (error) {
        clearInterval(progressInterval);
        alert('Network error: ' + error.message);
    } finally {
        document.getElementById('trainingProgress').style.display = 'none';
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        loadModelInfo(); // Refresh model info
    }
});

function displayTrainingResults(result) {
    document.getElementById('trainingResults').style.display = 'block';
    
    const metrics = result.metrics;
    document.getElementById('trainAccuracy').textContent = (metrics.test_accuracy * 100).toFixed(2) + '%';
    document.getElementById('trainPrecision').textContent = (metrics.test_precision * 100).toFixed(2) + '%';
    document.getElementById('trainRecall').textContent = (metrics.test_recall * 100).toFixed(2) + '%';
    document.getElementById('trainTime').textContent = metrics.training_time;
    
    // Load visualization images with cache busting
    if (result.plots) {
        const timestamp = new Date().getTime();
        if (result.plots.sample_predictions) {
            document.getElementById('samplePredictions').src = result.plots.sample_predictions + '?' + timestamp;
        }
        if (result.plots.confusion_matrix) {
            document.getElementById('confusionMatrix').src = result.plots.confusion_matrix + '?' + timestamp;
        }
    }
    
    // Check if goal achieved
    if (metrics.test_accuracy >= 0.95) {
        showSuccessAlert('🎉 Goal Achieved! Model accuracy is over 95%!');
    }
}

// Dataset info functions
async function loadDatasetInfo() {
    document.getElementById('datasetSpinner').style.display = 'block';
    document.getElementById('datasetInfo').style.display = 'none';
    
    try {
        const response = await fetch('/task2/dataset-info');
        const result = await response.json();
        
        document.getElementById('datasetSpinner').style.display = 'none';
        
        if (result.success) {
            displayDatasetInfo(result.statistics);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        document.getElementById('datasetSpinner').style.display = 'none';
        alert('Network error: ' + error.message);
    }
}

function displayDatasetInfo(stats) {
    document.getElementById('datasetInfo').style.display = 'block';
    
    const html = `
        <div class="row">
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6>Dataset Size</h6>
                        <p><strong>Training Samples:</strong> ${stats.training_samples.toLocaleString()}</p>
                        <p><strong>Test Samples:</strong> ${stats.test_samples.toLocaleString()}</p>
                        <p><strong>Total Samples:</strong> ${(stats.training_samples + stats.test_samples).toLocaleString()}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6>Image Properties</h6>
                        <p><strong>Image Shape:</strong> ${stats.image_shape.join('x')}</p>
                        <p><strong>Number of Classes:</strong> ${stats.num_classes}</p>
                        <p><strong>Pixel Range:</strong> ${stats.pixel_value_range.min.toFixed(2)} - ${stats.pixel_value_range.max.toFixed(2)}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6>Class Distribution (Training)</h6>
                        ${stats.class_distribution_train.map((count, digit) => `
                            <p class="mb-1"><strong>${digit}:</strong> ${count} samples</p>
                        `).join('')}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6>Class Distribution (Test)</h6>
                        ${stats.class_distribution_test.map((count, digit) => `
                            <p class="mb-1"><strong>${digit}:</strong> ${count} samples</p>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('datasetInfo').innerHTML = html;
}

// Model info functions
async function loadModelInfo() {
    try {
        const response = await fetch('/task2/model-info');
        const result = await response.json();
        
        const container = document.getElementById('currentModelInfo');
        
        if (result.success) {
            const info = result.model_info;
            container.innerHTML = `
                <p class="mb-1"><strong>Type:</strong> ${info.model_type}</p>
                <p class="mb-1"><strong>Input Shape:</strong> ${info.input_shape.join('x')}</p>
                <p class="mb-1"><strong>Classes:</strong> ${info.num_classes}</p>
                <p class="mb-1"><strong>Parameters:</strong> ${info.parameters ? info.parameters.toLocaleString() : 'Unknown'}</p>
                <p class="mb-0"><strong>Trained:</strong> ${info.training_date}</p>
            `;
        } else {
            container.innerHTML = '<p class="text-muted small">No model trained yet</p>';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Sample images functions
async function loadSampleImages(type) {
    const container = document.getElementById('sampleImagesContainer');
    container.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2">Loading samples...</p></div>';
    
    // In a real implementation, this would fetch actual sample images from the server
    // For now, we'll simulate loading
    setTimeout(() => {
        let html = `<h6>${type === 'train' ? 'Training' : 'Test'} Samples (First 20 images)</h6><div class="row">`;
        
        for (let i = 0; i < 20; i++) {
            // Create a simple placeholder - in real implementation, these would be actual MNIST images
            html += `
                <div class="col-3 col-sm-2 mb-3">
                    <div class="card sample-digit">
                        <div class="card-body text-center p-2">
                            <div class="bg-dark rounded" style="width: 50px; height: 50px; margin: 0 auto;"></div>
                            <small class="text-muted">Digit ${i % 10}</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    }, 1000);
}

// Utility functions
function showSuccessAlert(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show mt-3';
    alertDiv.innerHTML = `
        <strong>Success!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.getElementById('trainingResults').prepend(alertDiv);
}

// Load model info on page load
loadModelInfo();