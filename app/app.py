import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import config
from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.predict import TransactionPredictor
from src.explainability import ModelExplainer
from src.analytics import FinanceAnalytics
from src.anomaly_detection import SpendingAnomalyDetector

# Configure Streamlit page
st.set_page_config(
    page_title=config.app_params.get("title", "AI Personal Finance Advisor"),
    page_icon=config.app_params.get("page_icon", "💰"),
    layout=config.app_params.get("layout", "wide")
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .explanation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_services():
    """Cache model predictor and explainer instances."""
    predictor = TransactionPredictor()
    explainer = ModelExplainer(predictor)
    anomaly_detector = SpendingAnomalyDetector()
    return predictor, explainer, anomaly_detector

try:
    predictor, explainer, anomaly_detector = load_prediction_services()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Sidebar Navigation
st.sidebar.title("💰 Finance Advisor")
st.sidebar.caption("AI-Powered Transaction Classifier & Analytics")

nav_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / Overview",
        "🏷️ Categorize Transactions",
        "📊 Spending Analysis",
        "⚠️ Budget & Anomalies",
        "🔍 Model & Data Quality",
        "🛡️ Privacy & Advice Safety"
    ]
)

# Shared session state for uploaded data
if "user_df" not in st.session_state:
    st.session_state.user_df = None

# Header Banner
st.markdown('<div class="main-header">AI Personal Finance Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated transaction categorization, spending behavior analysis, and explainable financial insights.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: HOME / OVERVIEW
# -----------------------------------------------------------------------------
if nav_option == "🏠 Home / Overview":
    st.header("🏠 Executive Financial Overview")
    
    st.info("💡 **Getting Started**: Upload a transaction file in the **Categorize Transactions** tab or explore standard dataset analytics.")
    
    if st.session_state.user_df is not None:
        df_active = st.session_state.user_df
        st.success(f"Loaded User Transaction File with **{len(df_active):,}** transactions.")
    else:
        # Load sample from processed test set for preview
        try:
            test_path = config.raw_data_path.parent.parent / "processed" / "test.parquet"
            df_active = DataLoader.load_parquet(test_path).head(1000)
            st.caption("Displaying sample subset of verified benchmark dataset transactions.")
        except Exception:
            df_active = pd.DataFrame()

    if len(df_active) > 0:
        summary = FinanceAnalytics.calculate_summary(df_active)
        
        # Display Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{summary['total_transactions']:,}")
        with col2:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get("USD", next(iter(summary["by_currency"].values()), {}))
                st.metric("Total Income", f"${by_curr.get('total_income', 0.0):,.2f}")
            else:
                st.metric("Total Income", "N/A (No Amount Data)")
        with col3:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get("USD", next(iter(summary["by_currency"].values()), {}))
                st.metric("Total Expenses", f"${by_curr.get('total_expense', 0.0):,.2f}")
            else:
                st.metric("Total Expenses", "N/A (No Amount Data)")
        with col4:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get("USD", next(iter(summary["by_currency"].values()), {}))
                net = by_curr.get('net_cash_flow', 0.0)
                st.metric("Net Cash Flow", f"${net:,.2f}", delta=f"{net:,.2f}")
            else:
                st.metric("Net Cash Flow", "N/A (No Amount Data)")

        st.divider()

        # Category Breakdown Chart
        st.subheader("Category Distribution")
        cat_df = FinanceAnalytics.get_category_breakdown(df_active)
        
        if len(cat_df) > 0:
            fig = px.pie(
                cat_df,
                names=cat_df.columns[0],
                values="Percentage (%)" if "Percentage (%)" in cat_df.columns else cat_df.columns[1],
                title="Transaction Breakdown by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Transactions Preview")
        st.dataframe(df_active.head(20), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: CATEGORIZE TRANSACTIONS
# -----------------------------------------------------------------------------
elif nav_option == "🏷️ Categorize Transactions":
    st.header("🏷️ Transaction Categorization & File Upload")
    
    if not model_loaded:
        st.error(f"Prediction model artifact unavailable: {load_error}. Please train the model first.")
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Transaction Entry", "📁 CSV / Excel Batch Upload"])
    
    # Single prediction tab
    with tab_single:
        st.subheader("Predict Category for Single Transaction")
        desc_input = st.text_input(
            "Enter Transaction Description:",
            value="Netflix Subscription Renewal #4892",
            placeholder="e.g. Starbucks Store #1029, Salary Deposit, Uber Ride"
        )
        
        if st.button("Categorize Transaction", type="primary"):
            with st.spinner("Analyzing text and generating prediction..."):
                exp_res = explainer.explain_prediction(desc_input)
                
            st.success(f"Predicted Category: **{exp_res['predicted_category']}**")
            st.metric("Confidence Score", f"{exp_res['confidence'] * 100:.1f}%")
            
            # Top-k probabilities
            st.subheader("Top Category Probabilities")
            for item in exp_res["top_k"]:
                st.write(f"**{item['category']}**: {item['probability']*100:.1f}%")
                st.progress(float(item["probability"]))
                
            # Explainability
            st.subheader("💡 Model Explainability & Feature Attribution")
            st.markdown(f'<div class="explanation-box">{exp_res["explanation_summary"]}</div>', unsafe_allow_html=True)
            
            if exp_res["important_features"]:
                st.caption("Key words/n-grams driving this decision:")
                feat_df = pd.DataFrame(exp_res["important_features"])
                st.table(feat_df)

    # Batch upload tab
    with tab_batch:
        st.subheader("Upload Transaction File")
        uploaded_file = st.file_uploader(
            "Upload CSV file containing financial transactions", type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                raw_user_df = pd.read_csv(uploaded_file)
                is_valid, val_info, clean_user_df = DataValidator.validate_user_upload(raw_user_df)
                
                if not is_valid:
                    st.error("Validation Error: " + "; ".join(val_info["errors"]))
                else:
                    if val_info["warnings"]:
                        for w in val_info["warnings"]:
                            st.warning(w)
                            
                    st.info(f"Successfully validated **{val_info['valid_rows']}** rows.")
                    
                    if st.button("Process & Categorize Uploaded File", type="primary"):
                        with st.spinner("Categorizing transactions..."):
                            proc_df = predictor.predict_batch(clean_user_df)
                            
                            # Run anomaly detection if amount column present
                            if "amount" in proc_df.columns:
                                proc_df = anomaly_detector.detect_anomalies(proc_df)
                                
                            st.session_state.user_df = proc_df
                            st.success("Batch categorization complete! Results updated across dashboard.")
                            
                        st.subheader("Processed Transactions Output")
                        st.dataframe(proc_df, use_container_width=True)
                        
                        csv_data = proc_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Categorized CSV Report",
                            data=csv_data,
                            file_name="categorized_transactions_report.csv",
                            mime="text/csv"
                        )
            except Exception as ex:
                st.error(f"Error processing file: {ex}")

# -----------------------------------------------------------------------------
# TAB 3: SPENDING ANALYSIS
# -----------------------------------------------------------------------------
elif nav_option == "📊 Spending Analysis":
    st.header("📊 Spending Behavior & Financial Breakdown")
    
    df_active = st.session_state.user_df
    if df_active is None:
        try:
            test_path = config.raw_data_path.parent.parent / "processed" / "test.parquet"
            df_active = DataLoader.load_parquet(test_path).head(2000)
            st.caption("Displaying analysis on benchmark test dataset sample.")
        except Exception:
            df_active = pd.DataFrame()
            
    if len(df_active) > 0:
        cat_df = FinanceAnalytics.get_category_breakdown(df_active)
        
        st.subheader("Category-wise Spending Breakdown")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.dataframe(cat_df, use_container_width=True)
            
        with col_right:
            if len(cat_df) > 0:
                fig = px.bar(
                    cat_df,
                    x=cat_df.columns[0],
                    y="Percentage (%)" if "Percentage (%)" in cat_df.columns else cat_df.columns[1],
                    title="Spending Distribution (%)",
                    color=cat_df.columns[0],
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Monthly Trends if date exists
        st.subheader("Monthly Income vs Expense Trends")
        trends_df = FinanceAnalytics.get_monthly_trends(df_active)
        
        if len(trends_df) > 0:
            fig_trend = px.line(
                trends_df,
                x="YearMonth",
                y=["Income", "Expense", "Net_Cash_Flow"],
                title="Monthly Cash Flow Timeline",
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("ℹ️ Monthly timeline trend chart requires a valid 'date' and 'amount' column in uploaded transactions.")

# -----------------------------------------------------------------------------
# TAB 4: BUDGET & ANOMALIES
# -----------------------------------------------------------------------------
elif nav_option == "⚠️ Budget & Anomalies":
    st.header("⚠️ Budget Insights & Unusual Spending Flags")
    
    df_active = st.session_state.user_df
    if df_active is None:
        st.info("Please upload a transaction CSV file in the **Categorize Transactions** tab to run anomaly detection and recurring merchant analysis.")
    else:
        st.subheader("Unusual Spending Pattern Detection")
        if "amount" in df_active.columns:
            anom_df = anomaly_detector.detect_anomalies(df_active)
            unusual_records = anom_df[anom_df["is_unusual"] == True]
            
            st.metric("Flagged Unusual Transactions", f"{len(unusual_records):,}")
            
            if len(unusual_records) > 0:
                st.warning("⚠️ The following transactions were flagged as unusual spending patterns based on statistical amount thresholds or IsolationForest outliers.")
                st.dataframe(
                    unusual_records[["transaction_description", "amount", "predicted_category", "anomaly_reason"]],
                    use_container_width=True
                )
            else:
                st.success("✅ No unusual spending patterns detected in your uploaded dataset.")
        else:
            st.info("Amount data missing. Anomaly detection requires transaction amount values.")

        st.divider()
        
        st.subheader("Recurring Expenses & Subscription Detection")
        recurring_df = FinanceAnalytics.detect_recurring_transactions(df_active, min_count=2)
        if len(recurring_df) > 0:
            st.dataframe(recurring_df, use_container_width=True)
        else:
            st.info("No recurring transaction patterns detected.")

# -----------------------------------------------------------------------------
# TAB 5: MODEL & DATA QUALITY
# -----------------------------------------------------------------------------
elif nav_option == "🔍 Model & Data Quality":
    st.header("🔍 Dataset Quality Statistics & Model Benchmark Evaluation")
    
    reports_dir = config.reports_dir
    
    st.subheader("1. Dataset Statistics (Verified from Raw Data)")
    try:
        df_stats = pd.read_csv(reports_dir / "dataset_statistics.csv")
        st.table(df_stats)
    except Exception:
        st.info("Dataset statistics report not found.")

    st.subheader("2. Candidate Model Evaluation Benchmarks")
    try:
        df_models = pd.read_csv(reports_dir / "model_evaluation.csv")
        st.dataframe(df_models, use_container_width=True)
    except Exception:
        st.info("Model evaluation benchmark CSV not found.")

    col_cm, col_leak = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix (Winning Model)")
        cm_path = reports_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix on 30,000 Test Set", width="stretch")
        else:
            st.info("Confusion matrix image unavailable.")
            
    with col_leak:
        st.subheader("Data Leakage Audit Report")
        try:
            df_leak = pd.read_csv(reports_dir / "leakage_analysis.csv")
            st.dataframe(df_leak, use_container_width=True)
        except Exception:
            st.info("Leakage report unavailable.")

    st.subheader("3. Preprocessing Examples (Before vs After)")
    try:
        df_pre = pd.read_csv(reports_dir / "preprocessing_examples.csv").head(15)
        st.table(df_pre)
    except Exception:
        st.info("Preprocessing examples unavailable.")

# -----------------------------------------------------------------------------
# TAB 6: PRIVACY & ADVICE SAFETY
# -----------------------------------------------------------------------------
elif nav_option == "🛡️ Privacy & Advice Safety":
    st.header("🛡️ Privacy, Security & Educational Guidance Safety")
    
    st.subheader("Educational Financial Advice Safety Disclaimer")
    st.warning("""
    **IMPORTANT NOTICE**: 
    This application is designed as an educational personal finance management tool. 
    It provides automated transaction categorization and spending analysis based on statistical machine-learning models.
    
    - **Not Licensed Advice**: This tool does **NOT** provide certified or personalized financial, investment, legal, or tax advice.
    - **No Return Guarantees**: Any budget observations are strictly educational spending insights.
    - **Model Predictions**: Categorization results are generated by machine learning and should be verified for sensitive accounting.
    """)
    
    st.subheader("Privacy & Security Architecture")
    st.info("""
    - **Local Processing**: Uploaded CSV files are processed locally within your application environment and session.
    - **Zero Data Harvesting**: No transaction descriptions, passwords, or personal banking credentials are saved to external cloud APIs or committed to public Git repositories.
    - **Environment Controls**: All secret configuration files (`.env`) and raw transaction files are strictly excluded via `.gitignore`.
    """)
