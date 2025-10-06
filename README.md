**AI-Driven Next-Generation Firewall**
An intelligent network security system that leverages Deep Learning to detect and classify malicious network traffic in real-time. Built with PyTorch for model training and Streamlit for interactive visualization.

**Overview**
Traditional rule-based firewalls fail to detect modern sophisticated attacks such as zero-day exploits, encrypted threats, and anomalous behaviors. This project implements an AI-powered Next-Generation Firewall that uses Deep Neural Networks to intelligently analyze network traffic patterns and identify malicious activities with high accuracy.
The system provides:
Real-time threat detection and classification
Interactive monitoring dashboard
Automated incident response and logging
Zero Trust security architecture implementation
Comprehensive analytics and reporting

**Features**

AI-Powered Detection: Deep Neural Network trained on CICIDS2017 dataset achieving 98%+ accuracy
Real-Time Analysis: Upload and analyze network traffic CSV files instantly
Multi-Class Classification: Detects 15 different attack types including:
DDoS, DoS variants (Hulk, GoldenEye, Slowloris, Slowhttptest)
Port Scanning
Brute Force attacks (FTP, SSH)
Web Attacks (SQL Injection, XSS, Brute Force)
Botnet activity
Infiltration attempts
Heartbleed vulnerability exploitation

**Interactive Dashboard**: Beautiful Streamlit interface with 5 monitoring tabs
**Zero Trust Architecture**: Continuous verification and adaptive access control
**Automated Logging**: Track all detected threats with timestamps and probabilities
**Customizable Thresholds**: Adjust detection sensitivity in real-time
**Export Capabilities**: Download incident logs as CSV files

**Quick Start**
bash# Setup
git clone https://github.com/yourusername/ai-ngfw.git
cd ai-ngfw
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

**Dataset**
This project uses the CICIDS2017 dataset from the Canadian Institute for Cybersecurity.
Dataset Statistics:
Total Flows: ~2.8 million
Features: 78 network flow characteristics
Classes: 15 (1 benign + 14 attack types)
Time Period: Monday-Friday (5 days)
Size: ~2GB (CSV format)

# Run pipeline
python preprocess.py      # 10-30 min
python train_model.py     # 30-90 min (CPU)
streamlit run app.py      # Open http://localhost:8501
Requirements

Python 3.8+
4GB RAM minimum
PyTorch, Streamlit, scikit-learn, pandas

**Model**
Architecture: 78 → 256 → 128 → 64 → 15 classes
Dataset: CICIDS2017 (2.8M flows)
Accuracy: 98.67% | Precision: 98.45% | Recall: 98.34%

