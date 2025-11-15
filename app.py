from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import sys

# ==================== YOUR MODEL CODE (FROM TKINTER) ====================
print("\n" + "="*60)
print("🏥 Loading AI Disease Prediction Model...")
print("="*60)

model_loaded = False
clf = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    
    # ---------- Label mapping (YOUR CODE) ----------
    id2label = {
        "LABEL_0": "Acne",
        "LABEL_1": "AIDS",
        "LABEL_2": "Alcoholic hepatitis",
        "LABEL_3": "Allergy",
        "LABEL_4": "Arthritis",
        "LABEL_5": "Asthma",
        "LABEL_6": "Bronchitis",
        "LABEL_7": "Cervical Spondylosis",
        "LABEL_8": "Chicken pox",
        "LABEL_9": "Chronic cholestasis",
        "LABEL_10": "Common Cold",
        "LABEL_11": "COVID-19",
        "LABEL_12": "Dengue",
        "LABEL_13": "Diabetes",
        "LABEL_14": "Drug Reaction",
        "LABEL_15": "Fungal Infection",
        "LABEL_16": "Gastroenteritis",
        "LABEL_17": "GERD",
        "LABEL_18": "Heart Attack",
        "LABEL_19": "Hepatitis A",
        "LABEL_20": "Hepatitis B",
        "LABEL_21": "Hepatitis C",
        "LABEL_22": "Hepatitis D",
        "LABEL_23": "Hepatitis E",
        "LABEL_24": "Hypertension",
        "LABEL_25": "Hyperthyroidism",
        "LABEL_26": "Hypoglycemia",
        "LABEL_27": "Hypothyroidism",
        "LABEL_28": "Impetigo",
        "LABEL_29": "Jaundice",
        "LABEL_30": "Malaria",
        "LABEL_31": "Migraine",
        "LABEL_32": "Osteoarthritis",
        "LABEL_33": "Paralysis (Brain Hemorrhage)",
        "LABEL_34": "Peptic ulcer disease",
        "LABEL_35": "Pneumonia",
        "LABEL_36": "Psoriasis",
        "LABEL_37": "Tuberculosis",
        "LABEL_38": "Typhoid",
        "LABEL_39": "Urinary Tract Infection",
        "LABEL_40": "Varicose veins"
    }
    
    # ---------- Load model (YOUR CODE) ----------
    model_name = "shanover/symps_disease_bert_v3_c41"
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")
    print("Downloading model files from Hugging Face...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer loaded")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("✅ Model loaded")
    
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
    print("✅ Pipeline created")
    
    model_loaded = True
    print("\n✅ MODEL READY - Website can now predict diseases!")
    
except ImportError as e:
    print(f"\n❌ Missing libraries: {e}")
    print("\nPlease install:")
    print("  pip install transformers torch")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("="*60 + "\n")

# Storage for prediction history
prediction_history = []

# ==================== PREDICTION FUNCTION (YOUR CODE) ====================
def predict_disease(symptoms_text):
    """
    Predict disease from symptoms - EXACT same logic as your tkinter app
    """
    if not symptoms_text.strip():
        return {"error": "Please enter symptoms!"}
    
    try:
        # Use your pipeline (same as tkinter)
        results = clf(symptoms_text)[0]
        
        # Format top 5 predictions (same as tkinter)
        predictions = []
        for item in results[:5]:
            label = item["label"]
            score = item["score"]
            disease_name = id2label[label]
            
            predictions.append({
                'disease': disease_name,
                'confidence': float(score)
            })
        
        return {'predictions': predictions}
        
    except Exception as e:
        return {"error": str(e)}

# ==================== WEB SERVER HTML ====================
HTML = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Monitoring System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-primary: #f5f6fa;
            --bg-secondary: #ffffff;
            --bg-sidebar: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            --text-primary: #333333;
            --text-secondary: #7f8c8d;
            --text-tertiary: #2c3e50;
            --border-color: #e0e0e0;
            --card-shadow: rgba(0,0,0,0.08);
            --card-shadow-hover: rgba(0,0,0,0.15);
            --table-hover: #f8f9fa;
            --result-bg: #f8f9fa;
            --input-bg: #ffffff;
            --stat-text: #2c3e50;
        }

        [data-theme="dark"] {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-sidebar: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --text-tertiary: #ffffff;
            --border-color: #2d3748;
            --card-shadow: rgba(0,0,0,0.3);
            --card-shadow-hover: rgba(0,0,0,0.5);
            --table-hover: #1f2937;
            --result-bg: #1f2937;
            --input-bg: #2d3748;
            --stat-text: #eaeaea;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        /* Sidebar Navigation */
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 260px;
            height: 100vh;
            background: var(--bg-sidebar);
            padding: 20px 0;
            box-shadow: 2px 0 10px var(--card-shadow);
            z-index: 1000;
        }
        
        .logo {
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 40px;
            padding: 0 20px;
        }
        
        .logo i {
            display: block;
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .nav-menu {
            list-style: none;
        }
        
        .nav-item {
            margin: 5px 0;
        }
        
        .nav-link {
            display: flex;
            align-items: center;
            padding: 15px 25px;
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            transition: all 0.3s;
            border-left: 4px solid transparent;
            cursor: pointer;
        }
        
        .nav-link:hover,
        .nav-link.active {
            background: rgba(255,255,255,0.1);
            color: white;
            border-left-color: white;
        }
        
        .nav-link i {
            margin-right: 15px;
            font-size: 20px;
            width: 25px;
        }
        
        /* Dark Mode Toggle */
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 30px;
            z-index: 1001;
        }
        
        .theme-toggle-btn {
            background: var(--bg-secondary);
            border: 2px solid var(--border-color);
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            color: var(--text-primary);
            box-shadow: 0 2px 10px var(--card-shadow);
            transition: all 0.3s;
        }
        
        .theme-toggle-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 5px 20px var(--card-shadow-hover);
        }
        
        .theme-icon {
            display: none;
        }
        
        .theme-icon.active {
            display: block;
        }
        
        /* Main Content */
        .main-content {
            margin-left: 260px;
            padding: 30px;
            padding-top: 80px;
            min-height: 100vh;
        }
        
        .page {
            display: none;
        }
        
        .page.active {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .page-header {
            margin-bottom: 30px;
        }
        
        .page-header h1 {
            font-size: 32px;
            color: var(--text-tertiary);
            margin-bottom: 10px;
        }
        
        .page-header p {
            color: var(--text-secondary);
            font-size: 16px;
        }
        
        /* Cards */
        .card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px var(--card-shadow);
            margin-bottom: 25px;
        }
        
        .card-header {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text-tertiary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Status Box */
        .status-box {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .status-box i {
            font-size: 24px;
        }
        
        .status-box.success {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        
        .status-box.error {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        [data-theme="dark"] .status-box.success {
            background: #1e4620;
            color: #7ef07e;
            border-color: #2d5a2f;
        }
        
        [data-theme="dark"] .status-box.error {
            background: #5c1f1f;
            color: #ff6b6b;
            border-color: #7d2a2a;
        }
        
        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px var(--card-shadow);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px var(--card-shadow-hover);
        }
        
        .stat-info h3 {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 10px;
        }
        
        .stat-info p {
            font-size: 28px;
            font-weight: bold;
            color: var(--stat-text);
        }
        
        .stat-icon {
            width: 60px;
            height: 60px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            color: white;
        }
        
        .stat-icon.purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .stat-icon.blue { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }
        .stat-icon.green { background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%); }
        .stat-icon.orange { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); }
        
        /* Examples */
        .examples {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
        }
        
        [data-theme="dark"] .examples {
            background: linear-gradient(135deg, #1a2332 0%, #2d1b3d 100%);
        }
        
        .examples h3 {
            color: #1976d2;
            font-size: 16px;
            margin-bottom: 15px;
        }
        
        [data-theme="dark"] .examples h3 {
            color: #90caf9;
        }
        
        .example {
            padding: 12px 15px;
            margin: 8px 0;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid transparent;
            color: #333;
        }
        
        [data-theme="dark"] .example {
            background: #2d3748;
            color: #eaeaea;
        }
        
        .example:hover {
            background: #f8f9fa;
            border-left-color: #667eea;
            transform: translateX(5px);
        }
        
        [data-theme="dark"] .example:hover {
            background: #374151;
        }
        
        /* Form Elements */
        label {
            display: block;
            margin-bottom: 10px;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 16px;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid var(--border-color);
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            margin-bottom: 20px;
            background: var(--input-bg);
            color: var(--text-primary);
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        /* Results */
        .result-section {
            margin-top: 30px;
            display: none;
        }
        
        .result-section.show {
            display: block;
        }
        
        .primary-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .primary-result h3 {
            font-size: 24px;
            margin-bottom: 15px;
            color: white;
        }
        
        .confidence-bar {
            background: rgba(255,255,255,0.3);
            height: 10px;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            background: white;
            height: 100%;
            transition: width 0.5s ease;
        }
        
        .predictions-list h3 {
            color: var(--text-tertiary);
            margin-bottom: 15px;
            font-size: 20px;
        }
        
        .prediction-item {
            padding: 15px;
            margin: 10px 0;
            background: var(--bg-secondary);
            border-radius: 10px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px var(--card-shadow);
        }
        
        .prediction-item strong {
            color: var(--text-tertiary);
            font-size: 18px;
            display: block;
            margin-bottom: 5px;
        }
        
        .prediction-item small {
            color: var(--text-secondary);
        }
        
        .prediction-bar {
            background: var(--border-color);
            height: 6px;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .prediction-bar-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
        }
        
        /* Loader */
        .loader {
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid var(--border-color);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Alert */
        .alert-box {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            border-left: 5px solid;
            display: flex;
            align-items: start;
            gap: 15px;
        }
        
        .alert-info {
            background: #e3f2fd;
            border-color: #2196F3;
            color: #0d47a1;
        }
        
        [data-theme="dark"] .alert-info {
            background: #1a2332;
            color: #90caf9;
        }
        
        .warning {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            color: #856404;
        }
        
        [data-theme="dark"] .warning {
            background: #5c4a1f;
            border-color: #b8860b;
            color: #ffd966;
        }
        
        /* Disease List */
        .disease-list {
            columns: 2;
            column-gap: 20px;
        }
        
        .disease-list div {
            padding: 8px 0;
            color: var(--text-secondary);
            break-inside: avoid;
        }
        
        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }
        
        th {
            background: var(--result-bg);
            font-weight: 600;
            color: var(--text-tertiary);
        }
        
        tr:hover {
            background: var(--table-hover);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                width: 70px;
            }
            
            .logo-text,
            .nav-link span {
                display: none;
            }
            
            .main-content {
                margin-left: 70px;
                padding: 20px;
                padding-top: 80px;
            }
            
            .disease-list {
                columns: 1;
            }
            
            .theme-toggle {
                top: 10px;
                right: 10px;
            }
        }
    </style>
</head>
<body>
    <!-- Dark Mode Toggle -->
    <div class="theme-toggle">
        <button class="theme-toggle-btn" onclick="toggleTheme()" aria-label="Toggle dark mode">
            <i class="fas fa-moon theme-icon" id="darkIcon"></i>
            <i class="fas fa-sun theme-icon active" id="lightIcon"></i>
        </button>
    </div>
    
    <!-- Sidebar Navigation -->
    <aside class="sidebar">
        <div class="logo">
            <i class="fas fa-heartbeat"></i>
            <div class="logo-text">Health Monitor</div>
        </div>
        <ul class="nav-menu">
            <li class="nav-item">
                <a class="nav-link active" onclick="showPage('overview')">
                    <i class="fas fa-home"></i>
                    <span>Overview</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" onclick="showPage('predict')">
                    <i class="fas fa-stethoscope"></i>
                    <span>Predict Disease</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" onclick="showPage('history')">
                    <i class="fas fa-history"></i>
                    <span>History</span>
                </a>
            </li>
        </ul>
    </aside>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Overview Page -->
        <div id="overview" class="page active">
            <div class="page-header">
                <h1>🏥 Health Monitoring Overview</h1>
                <p>AI-Powered Disease Prediction System</p>
            </div>
            
            <div class="status-box MODEL_STATUS_CLASS">
                <i class="fas MODEL_STATUS_ICON"></i>
                <div>
                    <strong>MODEL_STATUS_TEXT</strong>
                    <div style="font-size: 14px; margin-top: 5px;">MODEL_STATUS_DETAIL</div>
                </div>
            </div>
            
            <div class="alert-box alert-info">
                <i class="fas fa-info-circle"></i>
                <div>
                    <strong>Welcome to Health Monitoring System</strong><br>
                    Our AI analyzes symptoms to predict potential diseases from 41 different conditions.
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-info">
                        <h3>Total Predictions</h3>
                        <p id="total-predictions">0</p>
                    </div>
                    <div class="stat-icon purple">
                        <i class="fas fa-chart-bar"></i>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-info">
                        <h3>System Accuracy</h3>
                        <p>92%</p>
                    </div>
                    <div class="stat-icon blue">
                        <i class="fas fa-bullseye"></i>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-info">
                        <h3>Diseases Detected</h3>
                        <p>41</p>
                    </div>
                    <div class="stat-icon green">
                        <i class="fas fa-viruses"></i>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-info">
                        <h3>Model Version</h3>
                        <p>3.0</p>
                    </div>
                    <div class="stat-icon orange">
                        <i class="fas fa-code-branch"></i>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-viruses"></i> All Detectable Diseases</span>
                </div>
                <div class="disease-list" id="disease-list">
                    <div>Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-lightbulb"></i> About This System</span>
                </div>
                <p style="line-height: 1.8; color: var(--text-secondary);">
                    This Health Monitoring System uses advanced BERT-based deep learning to analyze symptom descriptions 
                    and predict potential diseases. The model has been trained on comprehensive medical datasets and can 
                    identify 41 different health conditions with high accuracy. Simply describe your symptoms in the 
                    "Predict Disease" section to receive instant AI-powered health analysis.
                </p>
            </div>
        </div>
        
        <!-- Predict Page -->
        <div id="predict" class="page">
            <div class="page-header">
                <h1>🩺 Disease Prediction</h1>
                <p>Enter your symptoms for AI-powered health analysis</p>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-comment-medical"></i> Symptom Input</span>
                </div>
                
                <div class="examples">
                    <h3><i class="fas fa-lightbulb"></i> Quick Examples - Click to Try:</h3>
                    <div class="example" onclick="fillExample('I have high fever for 3 days, severe headache, muscle pain, weakness and dry cough')">
                        🤒 <strong>Flu symptoms:</strong> High fever, headache, muscle pain, cough
                    </div>
                    <div class="example" onclick="fillExample('Runny nose, continuous sneezing, watery eyes, itchy throat and mild fever')">
                        🤧 <strong>Cold/Allergy:</strong> Runny nose, sneezing, watery eyes
                    </div>
                    <div class="example" onclick="fillExample('Excessive thirst, frequent urination, extreme fatigue, blurred vision and unexplained weight loss')">
                        💉 <strong>Diabetes:</strong> Thirst, frequent urination, fatigue, blurred vision
                    </div>
                    <div class="example" onclick="fillExample('Severe chest pain, heavy sweating, shortness of breath, pain radiating to left arm and jaw')">
                        ❤️ <strong>Heart Attack:</strong> Chest pain, sweating, breathing difficulty
                    </div>
                    <div class="example" onclick="fillExample('Difficulty breathing, wheezing sound when exhaling, chest tightness and shortness of breath')">
                        🫁 <strong>Asthma:</strong> Breathing difficulty, wheezing, chest tightness
                    </div>
                </div>
                
                <label for="symptoms">
                    <i class="fas fa-notes-medical"></i> Describe your symptoms in detail:
                </label>
                <textarea 
                    id="symptoms" 
                    placeholder="Be as detailed as possible. Example: I have been experiencing high fever (102°F) for the past 3 days, along with severe headache, body aches, extreme weakness, and dry cough..."
                ></textarea>
                
                <button onclick="predict()" id="predictBtn">
                    <i class="fas fa-magic"></i> Analyze Symptoms
                </button>
                
                <div class="result-section" id="results">
                    <div id="resultContent"></div>
                </div>
            </div>
        </div>
        
        <!-- History Page -->
        <div id="history" class="page">
            <div class="page-header">
                <h1>📋 Prediction History</h1>
                <p>Your recent symptom analyses</p>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-history"></i> Recent Assessments</span>
                    <button class="btn btn-primary" onclick="loadHistory()">
                        <i class="fas fa-sync"></i> Refresh
                    </button>
                </div>
                <div id="historyContent">
                    <p style="text-align: center; color: var(--text-secondary); padding: 40px;">
                        No predictions yet. Try the Predict Disease page!
                    </p>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        const modelLoaded = MODEL_LOADED_VALUE;
        
        // Dark Mode Toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            document.getElementById('darkIcon').classList.toggle('active');
            document.getElementById('lightIcon').classList.toggle('active');
        }
        
        // Load saved theme
        window.addEventListener('load', () => {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            if (savedTheme === 'dark') {
                document.getElementById('darkIcon').classList.add('active');
                document.getElementById('lightIcon').classList.remove('active');
            }
            
            // Load diseases
            const diseases = DISEASE_LIST_JSON;
            const diseaseList = document.getElementById('disease-list');
            
            let html = '';
            diseases.forEach(disease => {
                html += `<div>• ${disease}</div>`;
            });
            diseaseList.innerHTML = html;
        });
        
        // Navigation
        function showPage(pageName) {
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });
            
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
            });
            
            document.getElementById(pageName).classList.add('active');
            event.target.closest('.nav-link').classList.add('active');
            
            if (pageName === 'history') {
                loadHistory();
            }
        }
        
        function fillExample(text) {
            document.getElementById('symptoms').value = text;
        }
        
        async function predict() {
            const symptoms = document.getElementById('symptoms').value.trim();
            
            if (!symptoms) {
                alert('⚠️ Please describe your symptoms');
                return;
            }
            
            if (!modelLoaded) {
                alert('❌ Model not loaded. Please check Overview page for instructions.');
                return;
            }
            
            const resultSection = document.getElementById('results');
            const resultContent = document.getElementById('resultContent');
            
            resultContent.innerHTML = `
                <div class="loader">
                    <div class="spinner"></div>
                    <p style="color: #667eea; font-size: 18px; font-weight: 600;">Analyzing symptoms...</p>
                    <small style="color: var(--text-secondary);">This may take a few seconds</small>
                </div>
            `;
            resultSection.classList.add('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symptoms: symptoms })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                const top = data.predictions[0];
                const confidence = (top.confidence * 100);
                
                let html = `
                    <div class="primary-result">
                        <h3>🎯 Most Likely: ${top.disease}</h3>
                        <div style="font-size: 18px; margin-top: 10px;">
                            Confidence: ${confidence.toFixed(2)}%
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                    
                    <div class="predictions-list">
                        <h3>📊 Top 5 Predictions:</h3>
                `;
                
                data.predictions.forEach((pred, index) => {
                    const conf = (pred.confidence * 100);
                    html += `
                        <div class="prediction-item">
                            <strong>${index + 1}. ${pred.disease}</strong>
                            <small>Confidence: ${conf.toFixed(4)}%</small>
                            <div class="prediction-bar">
                                <div class="prediction-bar-fill" style="width: ${conf}%"></div>
                            </div>
                        </div>
                    `;
                });
                
                html += `
                    </div>
                    <div class="warning">
                        <strong><i class="fas fa-exclamation-triangle"></i> Medical Disclaimer:</strong>
                        This AI analysis is for informational purposes only. Always consult a qualified healthcare professional.
                    </div>
                `;
                
                resultContent.innerHTML = html;
                
                // Update counter
                const current = parseInt(document.getElementById('total-predictions').textContent);
                document.getElementById('total-predictions').textContent = current + 1;
                
            } catch (error) {
                resultContent.innerHTML = `
                    <div style="padding: 25px; background: #ffebee; border-radius: 15px; border-left: 5px solid #f44336;">
                        <strong style="color: #c62828; font-size: 18px;">
                            <i class="fas fa-times-circle"></i> Error:
                        </strong>
                        <p style="margin-top: 10px; color: #666;">${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function loadHistory() {
            try {
                const response = await fetch('/history');
                const data = await response.json();
                
                const historyContent = document.getElementById('historyContent');
                
                if (!data.history || data.history.length === 0) {
                    historyContent.innerHTML = `
                        <p style="text-align: center; color: var(--text-secondary); padding: 40px;">
                            No predictions yet. Try the Predict Disease page!
                        </p>
                    `;
                    return;
                }
                
                let html = `
                    <table>
                        <thead>
                            <tr>
                                <th>Date & Time</th>
                                <th>Symptoms</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                data.history.slice().reverse().forEach(item => {
                    const time = new Date(item.timestamp).toLocaleString();
                    const symptoms = item.symptoms.length > 60 
                        ? item.symptoms.substring(0, 60) + '...' 
                        : item.symptoms;
                    const conf = (item.confidence * 100).toFixed(2);
                    
                    html += `
                        <tr>
                            <td>${time}</td>
                            <td>${symptoms}</td>
                            <td><strong>${item.prediction}</strong></td>
                            <td>${conf}%</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table>';
                historyContent.innerHTML = html;
                
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
    </script>
</body>
</html>"""

# ==================== WEB SERVER ====================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = HTML
            
            if model_loaded:
                html = html.replace('MODEL_STATUS_CLASS', 'success')
                html = html.replace('MODEL_STATUS_ICON', 'fa-check-circle')
                html = html.replace('MODEL_STATUS_TEXT', '✅ AI Model Loaded Successfully')
                html = html.replace('MODEL_STATUS_DETAIL', 'System ready to analyze symptoms')
                html = html.replace('MODEL_LOADED_VALUE', 'true')
            else:
                html = html.replace('MODEL_STATUS_CLASS', 'error')
                html = html.replace('MODEL_STATUS_ICON', 'fa-times-circle')
                html = html.replace('MODEL_STATUS_TEXT', '❌ Model Not Loaded')
                html = html.replace('MODEL_STATUS_DETAIL', 'Install: pip install transformers torch')
                html = html.replace('MODEL_LOADED_VALUE', 'false')
            
            diseases_json = json.dumps(list(id2label.values()))
            html = html.replace('DISEASE_LIST_JSON', diseases_json)
            
            self.wfile.write(html.encode())
        
        elif self.path == '/history':
            self.send_json({'history': prediction_history})
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                symptoms = data.get('symptoms', '')
                
                if not model_loaded or clf is None:
                    self.send_json({'error': 'Model not loaded'}, 500)
                    return
                
                print(f"\n🔍 Analyzing: {symptoms[:60]}...")
                
                result = predict_disease(symptoms)
                
                if 'error' in result:
                    self.send_json(result, 400)
                    return
                
                prediction_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'symptoms': symptoms,
                    'prediction': result['predictions'][0]['disease'],
                    'confidence': result['predictions'][0]['confidence']
                })
                
                if len(prediction_history) > 50:
                    prediction_history.pop(0)
                
                print(f"✅ Top: {result['predictions'][0]['disease']} ({result['predictions'][0]['confidence']:.4f})")
                
                self.send_json(result)
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.send_json({'error': str(e)}, 500)
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        if "POST /predict" in format % args:
            return
        print(f"[{self.date_time_string()}] {format % args}")

if __name__ == '__main__':
    PORT = 5000
    
    print("\n" + "="*60)
    print("🌐 HEALTH MONITORING SYSTEM")
    print("="*60)
    print(f"URL: http://localhost:{PORT}")
    print(f"Model: {'✅ LOADED' if model_loaded else '❌ NOT LOADED'}")
    if model_loaded:
        print(f"Diseases: {len(id2label)}")
        print("✅ Ready to predict!")
    else:
        print("\n⚠️  Install: pip install transformers torch")
    print("="*60)
    print(f"\n🚀 Open: http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server = HTTPServer(('', PORT), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped\n")
