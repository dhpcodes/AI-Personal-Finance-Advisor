
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import config
from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.predict import TransactionPredictor
from src.explainability import ModelExplainer
from src.analytics import FinanceAnalytics
from src.anomaly_detection import SpendingAnomalyDetector
from src.grounded_llm import GroundedFinancialAssistant


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title=config.app_params.get(
        "title", "AI Personal Finance Advisor"
    ),
    page_icon=config.app_params.get("page_icon", "💰"),
    layout=config.app_params.get("layout", "wide"),
)


# =============================================================================
# CUSTOM UI STYLING
# =============================================================================
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2rem;
            font-weight: 750;
            color: #172B4D;
            margin-bottom: 0.4rem;
        }

        .sub-header {
            font-size: 1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background-color: #F8FAFC;
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            text-align: center;
        }

        .explanation-box {
            background-color: #F0F9FF;
            border-left: 4px solid #0284C7;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }

        .evidence-card {
            background-color: #F8FAFC;
            border-left: 4px solid #10B981;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* Neutral input borders */
        div[data-baseweb="input"] > div {
            border-color: #CBD5E1;
            border-radius: 8px;
        }

        div[data-baseweb="input"] > div:focus-within {
            border-color: #176B67;
            box-shadow: 0 0 0 1px #176B67;
        }

        /* Text areas */
        div[data-baseweb="textarea"] > div {
            border-color: #CBD5E1;
            border-radius: 8px;
        }

        div[data-baseweb="textarea"] > div:focus-within {
            border-color: #176B67;
            box-shadow: 0 0 0 1px #176B67;
        }

        /* Primary buttons */
        .stButton > button[kind="primary"] {
            background-color: #176B67;
            border-color: #176B67;
            color: white;
            border-radius: 8px;
            padding: 0.55rem 1rem;
            font-weight: 600;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #125653;
            border-color: #125653;
            color: white;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #F4F7FB;
        }

        /* Slightly softer dividers */
        hr {
            border-color: #E2E8F0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# LOAD MODEL AND SERVICES
# =============================================================================
@st.cache_resource
def load_prediction_services():
    """Load and cache the existing model and finance services."""
    predictor = TransactionPredictor()
    explainer = ModelExplainer(predictor)
    anomaly_detector = SpendingAnomalyDetector()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None

    assistant = GroundedFinancialAssistant(api_key=api_key)

    return predictor, explainer, anomaly_detector, assistant


try:
    predictor, explainer, anomaly_detector, assistant = (
        load_prediction_services()
    )
    model_loaded = True
    load_error = None
except Exception as e:
    model_loaded = False
    load_error = str(e)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("💰 Finance Advisor")
st.sidebar.caption("AI-Powered Transaction Classifier & Analytics")

nav_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / Overview",
        "🏷️ Categorize Transactions",
        "🤖 AI Financial Chatbot",
        "📊 Spending Analysis",
        "⚠️ Budget & Anomalies",
        "🔍 Model & Data Quality",
        "🛡️ Privacy & Advice Safety",
    ],
)


# =============================================================================
# SESSION STATE
# =============================================================================
if "user_df" not in st.session_state:
    st.session_state.user_df = None

if "messages" not in st.session_state:
    st.session_state.messages = []


if nav_option == "🤖 AI Financial Chatbot":
    st.sidebar.divider()

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# APPLICATION HEADER
# =============================================================================
st.markdown(
    '<div class="main-header">AI Personal Finance Advisor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    "Automated transaction categorization, spending behavior analysis, "
    "and explainable financial insights."
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER: LOAD A DEMO DATASET
# =============================================================================
def load_demo_data(rows=1000):
    """Load a sample from the existing processed test dataset."""
    test_path = (
        config.raw_data_path.parent.parent
        / "processed"
        / "test.parquet"
    )
    return DataLoader.load_parquet(test_path).head(rows)


# =============================================================================
# HOME / OVERVIEW
# =============================================================================
if nav_option == "🏠 Home / Overview":

    st.header("Executive Financial Overview")

    st.info(
        "Upload transactions in **Categorize Transactions** or explore "
        "the available benchmark dataset."
    )

    if st.session_state.user_df is not None:
        df_active = st.session_state.user_df
        st.success(
            f"Loaded user dataset with {len(df_active):,} transactions."
        )
    else:
        try:
            df_active = load_demo_data(1000)
            st.caption(
                "Showing a sample of the benchmark test dataset."
            )
        except Exception as e:
            df_active = pd.DataFrame()
            st.warning(f"Could not load demo data: {e}")

    if not df_active.empty:
        summary = FinanceAnalytics.calculate_summary(df_active)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Transactions",
                f"{summary['total_transactions']:,}",
            )

        with col2:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get(
                    "USD",
                    next(iter(summary["by_currency"].values()), {}),
                )
                st.metric(
                    "Total Income",
                    f"${by_curr.get('total_income', 0.0):,.2f}",
                )
            else:
                st.metric("Total Income", "N/A")

        with col3:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get(
                    "USD",
                    next(iter(summary["by_currency"].values()), {}),
                )
                st.metric(
                    "Total Expenses",
                    f"${by_curr.get('total_expense', 0.0):,.2f}",
                )
            else:
                st.metric("Total Expenses", "N/A")

        with col4:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get(
                    "USD",
                    next(iter(summary["by_currency"].values()), {}),
                )
                net = by_curr.get("net_cash_flow", 0.0)
                st.metric("Net Cash Flow", f"${net:,.2f}")
            else:
                st.metric("Net Cash Flow", "N/A")

        st.divider()
        st.subheader("Category Distribution")

        cat_df = FinanceAnalytics.get_category_breakdown(df_active)

        if not cat_df.empty:
            percentage_col = (
                "Percentage (%)"
                if "Percentage (%)" in cat_df.columns
                else cat_df.columns[1]
            )

            fig = px.pie(
                cat_df,
                names=cat_df.columns[0],
                values=percentage_col,
                title="Transaction Breakdown by Category",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Transactions")
        st.dataframe(
            df_active.head(20),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# CATEGORIZE TRANSACTIONS
# =============================================================================
elif nav_option == "🏷️ Categorize Transactions":

    st.header("Transaction Categorization")
    st.caption(
        "Predict a category for an individual transaction or upload a CSV "
        "to categorize multiple transactions."
    )

    if not model_loaded:
        st.error(
            f"Prediction model unavailable: {load_error}. "
            "Check that the existing model artifact is present."
        )
        st.stop()

    tab_single, tab_batch = st.tabs(
        [
            "Single Transaction Entry",
            "📁 CSV Batch Upload",
        ]
    )

    with tab_single:
        st.subheader("Predict a Transaction Category")

        desc_input = st.text_input(
            "Transaction description",
            placeholder=(
                "e.g. Netflix Subscription, Grocery Store, Salary Deposit"
            ),
        )

        if st.button("Categorize Transaction", type="primary"):

            if not desc_input.strip():
                st.warning("Enter a transaction description first.")
            else:
                with st.spinner("Analyzing transaction..."):
                    exp_res = explainer.explain_prediction(desc_input)

                st.success(
                    "Predicted category: "
                    f"**{exp_res['predicted_category']}**"
                )

                st.metric(
                    "Confidence",
                    f"{exp_res['confidence'] * 100:.1f}%",
                )

                st.subheader("Top Category Probabilities")

                for item in exp_res["top_k"]:
                    st.write(
                        f"**{item['category']}** — "
                        f"{item['probability'] * 100:.1f}%"
                    )
                    st.progress(float(item["probability"]))

                st.subheader("Model Explanation")

                st.markdown(
                    '<div class="explanation-box">'
                    f"{exp_res['explanation_summary']}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if exp_res["important_features"]:
                    st.caption("Important words or phrases")
                    st.dataframe(
                        pd.DataFrame(exp_res["important_features"]),
                        use_container_width=True,
                        hide_index=True,
                    )

    with tab_batch:
        st.subheader("Upload Transactions")
        st.write(
            "Upload a CSV containing transaction descriptions and any "
            "required fields accepted by your data validator."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                raw_user_df = pd.read_csv(uploaded_file)

                is_valid, val_info, clean_user_df = (
                    DataValidator.validate_user_upload(raw_user_df)
                )

                if not is_valid:
                    st.error(
                        "Validation error: "
                        + "; ".join(val_info["errors"])
                    )
                else:
                    for warning in val_info["warnings"]:
                        st.warning(warning)

                    st.success(
                        f"Validated {val_info['valid_rows']:,} rows."
                    )

                    if st.button(
                        "Process & Categorize File",
                        type="primary",
                    ):
                        with st.spinner("Categorizing transactions..."):
                            proc_df = predictor.predict_batch(
                                clean_user_df
                            )

                            if "amount" in proc_df.columns:
                                proc_df = (
                                    anomaly_detector.detect_anomalies(
                                        proc_df
                                    )
                                )

                            st.session_state.user_df = proc_df

                        st.success(
                            "Categorization complete. The uploaded "
                            "dataset is now available to other sections."
                        )

                        st.subheader("Processed Transactions")
                        st.dataframe(
                            proc_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "📥 Download Categorized CSV",
                            data=proc_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="categorized_transactions_report.csv",
                            mime="text/csv",
                        )

            except Exception as ex:
                st.error(f"Error processing file: {ex}")


# =============================================================================
# AI FINANCIAL CHATBOT
# =============================================================================
elif nav_option == "🤖 AI Financial Chatbot":

    st.header("AI Financial Assistant")
    st.caption(
        "Ask about spending, monthly cash flow, recurring expenses, "
        "forecasts, and transaction classifications."
    )

    if st.session_state.user_df is not None:
        active_df = st.session_state.user_df
        st.success(
            f"Active dataset: {len(active_df):,} transactions."
        )
    else:
        try:
            active_df = load_demo_data(1000)
            st.info(
                "Demo mode: using 1,000 benchmark transactions. Upload your "
                "own CSV in Categorize Transactions for personalized analysis."
            )
        except Exception as e:
            active_df = pd.DataFrame()
            st.warning(f"No dataset is available: {e}")

    st.divider()
    st.subheader("Suggested Questions")

    col1, col2, col3 = st.columns(3)
    selected_prompt = None

    with col1:
        if st.button(
            "Where am I spending the most?",
            use_container_width=True,
        ):
            selected_prompt = "Where am I spending the most money?"

        if st.button(
            "What are my recurring expenses?",
            use_container_width=True,
        ):
            selected_prompt = "What are my recurring expenses?"

    with col2:
        if st.button(
            "How much did I spend on food this month?",
            use_container_width=True,
        ):
            selected_prompt = "How much did I spend on food this month?"

        if st.button(
            "Why was Netflix classified as entertainment?",
            use_container_width=True,
        ):
            selected_prompt = (
                "Why was 'Netflix Subscription' classified as entertainment?"
            )

    with col3:
        if st.button(
            "Estimate next month's expenses",
            use_container_width=True,
        ):
            selected_prompt = "What could my expenses look like next month?"

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("details"):
                with st.expander(
                    "🔍 View Verified Calculation Details & Evidence"
                ):
                    st.json(message["details"])

    user_input = st.chat_input(
        "Ask a question about your transaction data..."
    )
    prompt_to_process = selected_prompt or user_input

    if prompt_to_process:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt_to_process,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing and verifying calculations..."):
                response_obj = assistant.process_query(
                    prompt_to_process,
                    active_df,
                    history=st.session_state.messages,
                )

                ans_text = response_obj["answer"]
                tool_res = response_obj["tool_result"]
                intent = response_obj["intent"]

                st.markdown(ans_text)

                if (
                    intent == "spending_breakdown"
                    and tool_res.get("status") == "success"
                ):
                    b_df = pd.DataFrame(
                        tool_res.get("category_breakdown", [])
                    )

                    if b_df.empty:
                        st.info("No category spending data is available.")
                    elif {"category", "total_amount"}.issubset(
                        b_df.columns
                    ):
                        b_df["total_amount"] = pd.to_numeric(
                            b_df["total_amount"],
                            errors="coerce",
                        )
                        b_df = b_df.dropna(
                            subset=["category", "total_amount"]
                        )

                        b_df = (
                            b_df.groupby(
                                "category",
                                as_index=False,
                            )["total_amount"]
                            .sum()
                        )
                        b_df = b_df[b_df["total_amount"] > 0].copy()

                        total_spending = b_df["total_amount"].sum()

                        if total_spending > 0:
                            b_df["percentage"] = (
                                b_df["total_amount"] / total_spending * 100
                            )
                            b_df = b_df.sort_values(
                                "total_amount",
                                ascending=False,
                            )

                            st.subheader("📊 Complete Spending Breakdown")
                            st.metric(
                                "Total Recorded Spending",
                                f"{total_spending:,.2f}",
                            )
                            st.caption(
                                "Totals cover the transactions returned by "
                                "the analysis; they may not represent a "
                                "particular month."
                            )

                            fig = px.bar(
                                b_df,
                                x="category",
                                y="total_amount",
                                text=b_df["total_amount"].map(
                                    lambda x: f"{x:,.0f}"
                                ),
                                hover_data={
                                    "total_amount": ":,.2f",
                                    "percentage": ":.2f",
                                },
                                title="Spending by Category",
                                labels={
                                    "category": "Category",
                                    "total_amount": "Amount",
                                    "percentage": "Share of Total (%)",
                                },
                                color="category",
                                color_discrete_sequence=(
                                    px.colors.qualitative.Set2
                                ),
                            )
                            fig.update_traces(
                                textposition="outside",
                                cliponaxis=False,
                            )
                            fig.update_layout(
                                xaxis_tickangle=-30,
                                showlegend=False,
                                margin=dict(
                                    l=20,
                                    r=20,
                                    t=60,
                                    b=100,
                                ),
                            )
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                            display_df = b_df[
                                [
                                    "category",
                                    "total_amount",
                                    "percentage",
                                ]
                            ].rename(
                                columns={
                                    "category": "Category",
                                    "total_amount": "Total Amount",
                                    "percentage": "Percentage (%)",
                                }
                            )
                            st.dataframe(
                                display_df.style.format(
                                    {
                                        "Total Amount": "{:,.2f}",
                                        "Percentage (%)": "{:.2f}%",
                                    }
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.warning(
                            "The spending results are missing required fields."
                        )

                elif (
                    intent == "monthly_cash_flow"
                    and tool_res.get("status") == "success"
                ):
                    m_df = pd.DataFrame(
                        tool_res.get("monthly_cash_flow", [])
                    )
                    if not m_df.empty:
                        fig = px.line(
                            m_df,
                            x="YearMonth",
                            y=["Income", "Expense", "Net_Cash_Flow"],
                            title="Monthly Cash Flow",
                            markers=True,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                elif (
                    intent == "recurring_expenses"
                    and tool_res.get("status") == "success"
                ):
                    rec_items = tool_res.get("recurring_expenses", [])
                    if rec_items:
                        st.dataframe(
                            pd.DataFrame(rec_items),
                            use_container_width=True,
                            hide_index=True,
                        )

                with st.expander(
                    "🔍 View Verified Calculation Details & Evidence"
                ):
                    st.json(tool_res)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ans_text,
                "details": tool_res,
            }
        )


# =============================================================================
# SPENDING ANALYSIS
# =============================================================================
elif nav_option == "📊 Spending Analysis":

    st.header("Spending Behavior & Financial Breakdown")

    df_active = st.session_state.user_df

    if df_active is None:
        try:
            df_active = load_demo_data(2000)
            st.caption("Showing a sample of the benchmark test dataset.")
        except Exception as e:
            df_active = pd.DataFrame()
            st.warning(f"Could not load analysis data: {e}")

    if not df_active.empty:
        cat_df = FinanceAnalytics.get_category_breakdown(df_active)

        st.subheader("Category-wise Spending Breakdown")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.dataframe(
                cat_df,
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            if not cat_df.empty:
                percentage_col = (
                    "Percentage (%)"
                    if "Percentage (%)" in cat_df.columns
                    else cat_df.columns[1]
                )
                fig = px.bar(
                    cat_df,
                    x=cat_df.columns[0],
                    y=percentage_col,
                    title="Spending Distribution (%)",
                    color=cat_df.columns[0],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        st.divider()
        st.subheader("Monthly Income vs Expense Trends")

        trends_df = FinanceAnalytics.get_monthly_trends(df_active)

        if not trends_df.empty:
            fig_trend = px.line(
                trends_df,
                x="YearMonth",
                y=["Income", "Expense", "Net_Cash_Flow"],
                title="Monthly Cash Flow Timeline",
                markers=True,
            )
            st.plotly_chart(
                fig_trend,
                use_container_width=True,
            )
        else:
            st.info(
                "Monthly trends require valid date and amount columns."
            )
    else:
        st.info("No transaction data is available for analysis.")


# =============================================================================
# BUDGET & ANOMALIES
# =============================================================================
elif nav_option == "⚠️ Budget & Anomalies":

    st.header("Budget Insights & Unusual Spending Flags")
    df_active = st.session_state.user_df

    if df_active is None:
        st.info(
            "Upload a transaction CSV in Categorize Transactions to run "
            "anomaly detection and recurring merchant analysis."
        )
    else:
        st.subheader("Unusual Spending Detection")

        if "amount" in df_active.columns:
            anom_df = anomaly_detector.detect_anomalies(df_active)

            if "is_unusual" in anom_df.columns:
                unusual_records = anom_df[
                    anom_df["is_unusual"] == True
                ]
            else:
                unusual_records = pd.DataFrame()

            st.metric(
                "Flagged Unusual Transactions",
                f"{len(unusual_records):,}",
            )

            if not unusual_records.empty:
                st.warning(
                    "These transactions were flagged by statistical "
                    "thresholds or anomaly detection. Review them manually."
                )
                display_cols = [
                    col
                    for col in [
                        "transaction_description",
                        "amount",
                        "predicted_category",
                        "anomaly_reason",
                    ]
                    if col in unusual_records.columns
                ]
                st.dataframe(
                    unusual_records[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "An unusual transaction is not automatically fraud."
                )
            else:
                st.success("No unusual transactions were flagged.")
        else:
            st.info("Anomaly detection requires transaction amount values.")

        st.divider()
        st.subheader("Recurring Expenses & Subscriptions")

        recurring_df = FinanceAnalytics.detect_recurring_transactions(
            df_active,
            min_count=2,
        )

        if not recurring_df.empty:
            st.dataframe(
                recurring_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No recurring transaction patterns detected.")


# =============================================================================
# MODEL & DATA QUALITY
# =============================================================================
elif nav_option == "🔍 Model & Data Quality":

    st.header("Dataset Quality & Model Evaluation")
    reports_dir = config.reports_dir

    st.subheader("Dataset Statistics")
    try:
        df_stats = pd.read_csv(
            reports_dir / "dataset_statistics.csv"
        )
        st.dataframe(
            df_stats,
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.info("Dataset statistics report not found.")

    st.subheader("Candidate Model Evaluation")
    try:
        df_models = pd.read_csv(
            reports_dir / "model_evaluation.csv"
        )
        st.dataframe(
            df_models,
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.info("Model evaluation report not found.")

    col_cm, col_leak = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm_path = reports_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(
                str(cm_path),
                caption="Confusion Matrix on Test Set",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix image unavailable.")

    with col_leak:
        st.subheader("Data Leakage Audit")
        try:
            df_leak = pd.read_csv(
                reports_dir / "leakage_analysis.csv"
            )
            st.dataframe(
                df_leak,
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.info("Leakage report unavailable.")

    st.subheader("Preprocessing Examples")
    try:
        df_pre = pd.read_csv(
            reports_dir / "preprocessing_examples.csv"
        ).head(15)
        st.dataframe(
            df_pre,
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.info("Preprocessing examples unavailable.")


# =============================================================================
# PRIVACY & ADVICE SAFETY
# =============================================================================
elif nav_option == "🛡️ Privacy & Advice Safety":

    st.header("Privacy, Security & Financial Advice Safety")

    st.subheader("Educational Financial Guidance")

    st.warning(
        """
        **Important notice**

        This application provides educational transaction categorization
        and spending analysis. It does not provide certified financial,
        investment, legal, or tax advice.

        - Predictions may be incorrect and should be reviewed.
        - Spending forecasts are estimates, not guarantees.
        - Verify important transactions before making financial decisions.
        """
    )

    st.subheader("Privacy & Security")

    st.info(
        """
        - Uploaded CSV files are processed by this application.
        - Avoid uploading banking passwords or unnecessary sensitive data.
        - Keep `.env` and raw transaction files out of public repositories.
        """
    )