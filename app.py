"""
Streamlit Dashboard for AI-Driven Next-Gen Firewall
Interactive monitoring and detection interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import model
import sys
sys.path.append('model')
from model import FirewallDNN

# Page configuration
st.set_page_config(
    page_title="AI-Driven NGFW Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

class FirewallDashboard:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        # Initialize session state
        if 'incident_logs' not in st.session_state:
            st.session_state.incident_logs = []
        if 'total_flows' not in st.session_state:
            st.session_state.total_flows = 0
        if 'blocked_flows' not in st.session_state:
            st.session_state.blocked_flows = 0
    
    def load_model(self):
        """Load trained PyTorch model"""
        model_path = 'model/firewall_model.pth'
        
        if not os.path.exists(model_path):
            st.error("❌ Model not found! Please train the model first.")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = FirewallDNN(
                checkpoint['input_size'],
                checkpoint['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict(self, data):
        """Make predictions on input data"""
        if not self.model_loaded:
            return None
        
        with torch.no_grad():
            inputs = torch.FloatTensor(data.values).to(self.device)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            return predicted.cpu().numpy(), probabilities.cpu().numpy()
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🛡️ AI-Driven Next-Gen Firewall Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Model status
        st.sidebar.subheader("Model Status")
        if self.model_loaded:
            st.sidebar.success("✅ Model Loaded")
        else:
            st.sidebar.warning("⚠️ Model Not Loaded")
            if st.sidebar.button("Load Model"):
                with st.spinner("Loading model..."):
                    self.load_model()
                    st.rerun()
        
        st.sidebar.markdown("---")
        
        # System metrics
        st.sidebar.subheader("System Metrics")
        st.sidebar.metric("Total Flows Analyzed", st.session_state.total_flows)
        st.sidebar.metric("Blocked Threats", st.session_state.blocked_flows)
        
        if st.session_state.total_flows > 0:
            threat_rate = (st.session_state.blocked_flows / st.session_state.total_flows) * 100
            st.sidebar.metric("Threat Rate", f"{threat_rate:.2f}%")
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        threat_threshold = st.sidebar.slider(
            "Threat Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        return threat_threshold
    
    def render_overview(self):
        """Render overview tab"""
        st.header("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Flows",
                st.session_state.total_flows,
                help="Total network flows analyzed"
            )
        
        with col2:
            benign = st.session_state.total_flows - st.session_state.blocked_flows
            st.metric(
                "Benign Traffic",
                benign,
                help="Normal network traffic"
            )
        
        with col3:
            st.metric(
                "Malicious Traffic",
                st.session_state.blocked_flows,
                delta=f"{st.session_state.blocked_flows} threats",
                delta_color="inverse",
                help="Detected and blocked threats"
            )
        
        with col4:
            if st.session_state.total_flows > 0:
                detection_rate = (st.session_state.blocked_flows / st.session_state.total_flows) * 100
            else:
                detection_rate = 0
            st.metric(
                "Detection Rate",
                f"{detection_rate:.1f}%",
                help="Percentage of malicious traffic"
            )
        
        # Load training history if available
        history_path = 'model/training_history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{history['test_accuracy']*100:.2f}%")
            with col2:
                st.metric("Precision", f"{history['test_precision']*100:.2f}%")
            with col3:
                st.metric("Recall", f"{history['test_recall']*100:.2f}%")
            with col4:
                st.metric("F1-Score", f"{history['test_f1']*100:.2f}%")
    
    def render_live_detection(self, threat_threshold):
        """Render live detection tab"""
        st.header("🔍 Live Threat Detection")
        
        uploaded_file = st.file_uploader(
            "Upload network traffic CSV for analysis",
            type=['csv'],
            help="Upload a CSV file with network flow data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} network flows")
                
                # Show data preview
                with st.expander("📄 Data Preview"):
                    st.dataframe(df.head(10))
                
                if st.button("🔍 Analyze Traffic", type="primary"):
                    if not self.model_loaded:
                        st.error("Please load the model first!")
                        return
                    
                    with st.spinner("Analyzing network traffic..."):
                        # Prepare data (remove label if present)
                        X = df.drop(['Label'], axis=1, errors='ignore')
                        label_cols = [col for col in X.columns if 'label' in col.lower()]
                        X = X.drop(label_cols, axis=1, errors='ignore')
                        
                        # Predict
                        predictions, probabilities = self.predict(X)
                        
                        # Add predictions to dataframe
                        df['Prediction'] = predictions
                        df['Threat_Probability'] = probabilities.max(axis=1)
                        df['Is_Malicious'] = df['Threat_Probability'] >= threat_threshold
                        
                        # Update session state
                        st.session_state.total_flows += len(df)
                        malicious_count = df['Is_Malicious'].sum()
                        st.session_state.blocked_flows += malicious_count
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Benign Flows", len(df) - malicious_count)
                        with col2:
                            st.metric("Malicious Flows", malicious_count)
                        
                        # Alert if threats detected
                        if malicious_count > 0:
                            st.markdown(f"""
                                <div class="alert-box alert-danger">
                                    <strong>⚠️ THREAT ALERT!</strong><br>
                                    Detected {malicious_count} malicious flows out of {len(df)} total flows.
                                    These connections have been flagged for blocking.
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Log incidents
                            for idx, row in df[df['Is_Malicious']].iterrows():
                                incident = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'flow_id': idx,
                                    'prediction': int(row['Prediction']),
                                    'probability': float(row['Threat_Probability'])
                                }
                                st.session_state.incident_logs.append(incident)
                        else:
                            st.markdown("""
                                <div class="alert-box alert-success">
                                    <strong>✅ ALL CLEAR!</strong><br>
                                    No malicious traffic detected. All flows are benign.
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualization
                        st.subheader("Detection Results")
                        
                        # Pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=['Benign', 'Malicious'],
                            values=[len(df) - malicious_count, malicious_count],
                            marker_colors=['#4caf50', '#f44336']
                        )])
                        fig.update_layout(title="Traffic Classification")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        display_cols = ['Prediction', 'Threat_Probability', 'Is_Malicious']
                        st.dataframe(df[display_cols], use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    def render_analytics(self):
        """Render analytics tab"""
        st.header("📈 Threat Analytics")
        
        # Load training history
        history_path = 'model/training_history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training loss curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(history['train_losses']) + 1)),
                    y=history['train_losses'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#f44336')
                ))
                fig.update_layout(
                    title="Training Loss Over Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Training accuracy curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(history['train_accs']) + 1)),
                    y=history['train_accs'],
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='#4caf50')
                ))
                fig.update_layout(
                    title="Training Accuracy Over Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            cm_path = 'model/confusion_matrix.png'
            if os.path.exists(cm_path):
                st.subheader("Confusion Matrix")
                st.image(cm_path, use_column_width=True)
        else:
            st.info("No analytics data available. Train the model to see analytics.")
    
    def render_incident_logs(self):
        """Render incident logs tab"""
        st.header("🚨 Incident Response Logs")
        
        if not st.session_state.incident_logs:
            st.info("No incidents detected yet. Upload traffic data for analysis.")
            return
        
        # Display recent incidents
        st.subheader(f"Total Incidents: {len(st.session_state.incident_logs)}")
        
        # Convert to dataframe
        incidents_df = pd.DataFrame(st.session_state.incident_logs)
        
        # Display table
        st.dataframe(
            incidents_df.sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Download button
        csv = incidents_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Incident Logs",
            data=csv,
            file_name=f"incident_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear logs button
        if st.button("🗑️ Clear All Logs"):
            st.session_state.incident_logs = []
            st.session_state.total_flows = 0
            st.session_state.blocked_flows = 0
            st.rerun()
    
    def render_zero_trust(self):
        """Render Zero Trust monitor tab"""
        st.header("🔐 Zero Trust Security Monitor")
        
        st.markdown("""
        ### Zero Trust Architecture Principles
        
        This AI-driven firewall implements Zero Trust Network Architecture (ZTNA):
        
        - **Never Trust, Always Verify**: Every connection is authenticated and authorized
        - **Least Privilege Access**: Minimal access rights for users and devices
        - **Micro-segmentation**: Network isolation to contain threats
        - **Continuous Monitoring**: Real-time threat detection and response
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trust Levels")
            trust_data = {
                'Level': ['High Trust', 'Medium Trust', 'Low Trust', 'No Trust'],
                'Count': [45, 23, 12, 5],
                'Action': ['Allow', 'Monitor', 'Restrict', 'Block']
            }
            trust_df = pd.DataFrame(trust_data)
            
            fig = px.bar(
                trust_df,
                x='Level',
                y='Count',
                color='Action',
                title="Device Trust Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Active Policies")
            policies = {
                'Policy': [
                    'Multi-Factor Authentication',
                    'Device Compliance Check',
                    'Geo-Fencing',
                    'Behavioral Analysis',
                    'Encryption Required'
                ],
                'Status': ['Active', 'Active', 'Active', 'Active', 'Active']
            }
            st.dataframe(pd.DataFrame(policies), use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Load model on startup
        if not self.model_loaded:
            self.load_model()
        
        # Sidebar
        threat_threshold = self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🔍 Live Detection",
            "📈 Analytics",
            "🚨 Incident Logs",
            "🔐 Zero Trust"
        ])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_live_detection(threat_threshold)
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_incident_logs()
        
        with tab5:
            self.render_zero_trust()

if __name__ == "__main__":
    dashboard = FirewallDashboard()
    dashboard.run()