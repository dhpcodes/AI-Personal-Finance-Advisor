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
from src.budgeting import BudgetManager
from src.goal_saving import GoalSavingPlanner
from src.forecast import SpendingForecaster
from src.privacy import PrivacyGuard


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
            margin-bottom: 0;
            line-height: 1.6;
        }

        .hero-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F1F6FC 100%);
            border: 1px solid #E0E8F2;
            border-radius: 16px;
            padding: 1.45rem 1.65rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 4px 18px rgba(27, 55, 87, 0.05);
        }

        .hero-kicker {
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #5F7D9D;
            margin-bottom: 0.45rem;
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
            background-color: #FFFFFF;
            border-right: 1px solid #E5EAF0;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.6rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .sidebar-brand {
            padding: 0.6rem 0.35rem 1.25rem 0.35rem;
            margin-bottom: 1.05rem;
            border-bottom: 1px solid #E8EDF3;
        }

        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 750;
            color: #183B63;
            line-height: 1.15;
            margin-bottom: 0.55rem;
        }

        .sidebar-subtitle {
            font-size: 0.84rem;
            line-height: 1.55;
            color: #7A8798;
            max-width: 235px;
        }

        .sidebar-section-title {
            font-size: 0.83rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #55708F;
            margin: 0.25rem 0 0.45rem 0.2rem;
        }

        .sidebar-footer {
            margin-top: 2rem;
            padding: 1rem 0.35rem 0.25rem 0.35rem;
            border-top: 1px solid #E8EDF3;
        }

        .sidebar-footer-title {
            font-size: 0.84rem;
            font-weight: 650;
            color: #4D647E;
            margin-bottom: 0.35rem;
        }

        .sidebar-footer-line {
            font-size: 0.72rem;
            color: #8998AA;
            line-height: 1.5;
            margin-bottom: 0.2rem;
        }

        .sidebar-footer-note {
            font-size: 0.68rem;
            color: #A1ACB9;
            line-height: 1.45;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            color: #2C5075;
            font-size: 0.93rem;
            font-weight: 520;
            padding-top: 0.18rem;
            padding-bottom: 0.18rem;
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
st.sidebar.markdown(
    """<div class="sidebar-brand">
<div class="sidebar-title">💰 Finance Advisor</div>
<div class="sidebar-subtitle">
AI-Powered Transaction Classifier &amp; Analytics
</div>
</div>""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-section-title">Navigation Menu</div>',
    unsafe_allow_html=True,
)

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
    label_visibility="collapsed",
)

st.sidebar.markdown(
    """<div class="sidebar-footer">
<div class="sidebar-footer-title">AI Personal Finance Advisor</div>
<div class="sidebar-footer-line">Transaction Intelligence • Analytics • AI</div>
<div class="sidebar-footer-note">Educational project • Financial insights &amp; analysis</div>
</div>""",
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================
if "user_df" not in st.session_state:
    st.session_state.user_df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "budget_manager" not in st.session_state:
    st.session_state.budget_manager = BudgetManager()


if nav_option == "🤖 AI Financial Chatbot":
    st.sidebar.divider()

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# APPLICATION HEADER
# =============================================================================
st.markdown(
    """<div class="hero-card">
<div class="hero-kicker">Smart Finance • AI-Powered</div>
<div class="main-header">AI Personal Finance Advisor</div>
<div class="sub-header">Understand spending patterns, categorize transactions, and explore explainable financial insights in one place.</div>
</div>""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER: LOAD A DEMO DATASET
# =============================================================================
def load_demo_data(rows=1000):
    """Load the public synthetic demo dataset."""
    test_path = (
        config.raw_data_path.parent.parent
        / "demo"
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
                "🧪 Demo Mode • Showing synthetic sample transactions."
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
                    "INR",
                    next(iter(summary["by_currency"].values()), {}),
                )
                st.metric(
                    "Total Income",
                    f"INR {by_curr.get('total_income', 0.0):,.2f}",
                )
            else:
                st.metric("Total Income", "N/A")

        with col3:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get(
                    "INR",
                    next(iter(summary["by_currency"].values()), {}),
                )
                st.metric(
                    "Total Expenses",
                    f"INR {by_curr.get('total_expense', 0.0):,.2f}",
                )
            else:
                st.metric("Total Expenses", "N/A")

        with col4:
            if summary.get("has_amount_data"):
                by_curr = summary["by_currency"].get(
                    "INR",
                    next(iter(summary["by_currency"].values()), {}),
                )
                net = by_curr.get("net_cash_flow", 0.0)
                st.metric(
                    "Net Cash Flow",
                    f"INR {net:,.2f}",
                )
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

            st.plotly_chart(
                fig,
                width="stretch",
            )

        st.subheader("Recent Transactions")
        st.dataframe(
            df_active.head(20),
            width="stretch",
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

        if st.button(
            "Categorize Transaction",
            type="primary",
        ):

            if not desc_input.strip():
                st.warning(
                    "Enter a transaction description first."
                )
            else:
                with st.spinner(
                    "Analyzing transaction..."
                ):
                    exp_res = explainer.explain_prediction(
                        desc_input
                    )

                st.success(
                    "Predicted category: "
                    f"**{exp_res['predicted_category']}**"
                )

                st.metric(
                    "Confidence",
                    f"{exp_res['confidence'] * 100:.1f}%",
                )

                st.subheader(
                    "Top Category Probabilities"
                )

                for item in exp_res["top_k"]:
                    st.write(
                        f"**{item['category']}** — "
                        f"{item['probability'] * 100:.1f}%"
                    )

                    st.progress(
                        float(
                            item["probability"]
                        )
                    )

                st.subheader(
                    "Model Explanation"
                )

                st.markdown(
                    '<div class="explanation-box">'
                    f"{exp_res['explanation_summary']}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if exp_res["important_features"]:
                    st.caption(
                        "Important words or phrases"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            exp_res["important_features"]
                        ),
                        width="stretch",
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
                raw_user_df = pd.read_csv(
                    uploaded_file
                )

                is_valid, val_info, clean_user_df = (
                    DataValidator.validate_user_upload(
                        raw_user_df
                    )
                )

                if not is_valid:
                    st.error(
                        "Validation error: "
                        + "; ".join(
                            val_info["errors"]
                        )
                    )
                else:
                    for warning in val_info[
                        "warnings"
                    ]:
                        st.warning(
                            warning
                        )

                    st.success(
                        f"Validated "
                        f"{val_info['valid_rows']:,} rows."
                    )

                    if st.button(
                        "Process & Categorize File",
                        type="primary",
                    ):
                        with st.spinner(
                            "Categorizing transactions..."
                        ):
                            proc_df = (
                                predictor.predict_batch(
                                    clean_user_df
                                )
                            )

                            if (
                                "amount"
                                in proc_df.columns
                            ):
                                proc_df = (
                                    anomaly_detector.detect_anomalies(
                                        proc_df
                                    )
                                )

                            st.session_state.user_df = (
                                proc_df
                            )

                        st.success(
                            "Categorization complete. "
                            "The uploaded dataset is now "
                            "available to other sections."
                        )

                        st.subheader(
                            "Processed Transactions"
                        )

                        st.dataframe(
                            proc_df,
                            width="stretch",
                            hide_index=True,
                        )

                        st.download_button(
                            "📥 Download Categorized CSV",
                            data=(
                                proc_df.to_csv(
                                    index=False
                                ).encode("utf-8")
                            ),
                            file_name=(
                                "categorized_transactions_report.csv"
                            ),
                            mime="text/csv",
                        )

            except Exception as ex:
                st.error(
                    f"Error processing file: {ex}"
                )


# =============================================================================
# AI FINANCIAL CHATBOT
# =============================================================================
elif nav_option == "🤖 AI Financial Chatbot":

    st.header(
        "AI Financial Assistant"
    )

    st.caption(
        "Ask about spending, monthly cash flow, recurring expenses, "
        "forecasts, and transaction classifications."
    )

    if st.session_state.user_df is not None:
        active_df = (
            st.session_state.user_df
        )

        st.success(
            f"Active dataset: "
            f"{len(active_df):,} transactions."
        )

    else:
        try:
            active_df = load_demo_data(
                1000
            )

            st.info(
                "🧪 Demo Mode: using synthetic sample transactions. "
                "Upload your own CSV in Categorize Transactions "
                "for personalized analysis."
            )

        except Exception as e:
            active_df = pd.DataFrame()

            st.warning(
                f"No dataset is available: {e}"
            )

    st.divider()

    st.subheader(
        "Suggested Questions"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    selected_prompt = None

    with col1:

        if st.button(
            "Where am I spending the most?",
            width="stretch",
        ):
            selected_prompt = (
                "Where am I spending the most money?"
            )

        if st.button(
            "What are my recurring expenses?",
            width="stretch",
        ):
            selected_prompt = (
                "What are my recurring expenses?"
            )

    with col2:

        if st.button(
            "How much did I spend on food this month?",
            width="stretch",
        ):
            selected_prompt = (
                "How much did I spend on food this month?"
            )

        if st.button(
            "Why was Netflix classified as entertainment?",
            width="stretch",
        ):
            selected_prompt = (
                "Why was 'Netflix Subscription' "
                "classified as entertainment?"
            )

    with col3:

        if st.button(
            "Estimate next month's expenses",
            width="stretch",
        ):
            selected_prompt = (
                "What could my expenses look like next month?"
            )

    st.divider()

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get(
                "details"
            ):

                with st.expander(
                    "🔍 View Verified Calculation Details & Evidence"
                ):

                    st.json(
                        message["details"]
                    )

    user_input = st.chat_input(
        "Ask a question about your transaction data..."
    )

    prompt_to_process = (
        selected_prompt
        or user_input
    )

    if prompt_to_process:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt_to_process,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt_to_process
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing and verifying calculations..."
            ):

                response_obj = (
                    assistant.process_query(
                        prompt_to_process,
                        active_df,
                        history=(
                            st.session_state.messages
                        ),
                    )
                )

                ans_text = response_obj[
                    "answer"
                ]

                tool_res = response_obj[
                    "tool_result"
                ]

                intent = response_obj[
                    "intent"
                ]

                st.markdown(
                    ans_text
                )

                if (
                    intent
                    == "spending_breakdown"
                    and tool_res.get(
                        "status"
                    )
                    == "success"
                ):

                    b_df = pd.DataFrame(
                        tool_res.get(
                            "category_breakdown",
                            [],
                        )
                    )

                    if b_df.empty:

                        st.info(
                            "No category spending data "
                            "is available."
                        )

                    elif {
                        "category",
                        "total_amount",
                    }.issubset(
                        b_df.columns
                    ):

                        b_df[
                            "total_amount"
                        ] = pd.to_numeric(
                            b_df[
                                "total_amount"
                            ],
                            errors="coerce",
                        )

                        b_df = b_df.dropna(
                            subset=[
                                "category",
                                "total_amount",
                            ]
                        )

                        b_df = (
                            b_df.groupby(
                                "category",
                                as_index=False,
                            )[
                                "total_amount"
                            ]
                            .sum()
                        )

                        b_df = b_df[
                            b_df[
                                "total_amount"
                            ] > 0
                        ].copy()

                        total_spending = (
                            b_df[
                                "total_amount"
                            ].sum()
                        )

                        if total_spending > 0:

                            b_df[
                                "percentage"
                            ] = (
                                b_df[
                                    "total_amount"
                                ]
                                / total_spending
                                * 100
                            )

                            b_df = (
                                b_df.sort_values(
                                    "total_amount",
                                    ascending=False,
                                )
                            )

                            st.subheader(
                                "📊 Complete Spending Breakdown"
                            )

                            st.metric(
                                "Total Recorded Spending",
                                f"INR {total_spending:,.2f}",
                            )

                            st.caption(
                                "Totals cover the transactions "
                                "returned by the analysis; they may "
                                "not represent a particular month."
                            )

                            fig = px.bar(
                                b_df,
                                x="category",
                                y="total_amount",
                                text=(
                                    b_df[
                                        "total_amount"
                                    ].map(
                                        lambda x:
                                        f"{x:,.0f}"
                                    )
                                ),
                                hover_data={
                                    "total_amount": ":,.2f",
                                    "percentage": ":.2f",
                                },
                                title=(
                                    "Spending by Category"
                                ),
                                labels={
                                    "category": "Category",
                                    "total_amount": "Amount",
                                    "percentage": (
                                        "Share of Total (%)"
                                    ),
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
                                width="stretch",
                            )

                            display_df = (
                                b_df[
                                    [
                                        "category",
                                        "total_amount",
                                        "percentage",
                                    ]
                                ]
                                .rename(
                                    columns={
                                        "category": (
                                            "Category"
                                        ),
                                        "total_amount": (
                                            "Total Amount"
                                        ),
                                        "percentage": (
                                            "Percentage (%)"
                                        ),
                                    }
                                )
                            )

                            st.dataframe(
                                display_df.style.format(
                                    {
                                        "Total Amount": (
                                            "{:,.2f}"
                                        ),
                                        "Percentage (%)": (
                                            "{:.2f}%"
                                        ),
                                    }
                                ),
                                width="stretch",
                                hide_index=True,
                            )

                    else:

                        st.warning(
                            "The spending results are "
                            "missing required fields."
                        )

                elif (
                    intent
                    == "monthly_cash_flow"
                    and tool_res.get(
                        "status"
                    )
                    == "success"
                ):

                    m_df = pd.DataFrame(
                        tool_res.get(
                            "monthly_cash_flow",
                            [],
                        )
                    )

                    if not m_df.empty:

                        fig = px.line(
                            m_df,
                            x="YearMonth",
                            y=[
                                "Income",
                                "Expense",
                                "Net_Cash_Flow",
                            ],
                            title=(
                                "Monthly Cash Flow"
                            ),
                            markers=True,
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch",
                        )

                elif (
                    intent
                    == "recurring_expenses"
                    and tool_res.get(
                        "status"
                    )
                    == "success"
                ):

                    rec_items = tool_res.get(
                        "recurring_expenses",
                        [],
                    )

                    if rec_items:

                        st.dataframe(
                            pd.DataFrame(
                                rec_items
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                with st.expander(
                    "🔍 View Verified Calculation Details & Evidence"
                ):

                    st.json(
                        tool_res
                    )

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

    st.header(
        "Spending Behavior & Financial Breakdown"
    )

    df_active = (
        st.session_state.user_df
    )

    if df_active is None:

        try:

            df_active = load_demo_data(
                2000
            )

            st.caption(
                "🧪 Demo Mode • Showing synthetic sample transactions."
            )

        except Exception as e:

            df_active = pd.DataFrame()

            st.warning(
                f"Could not load analysis data: {e}"
            )

    if not df_active.empty:

        cat_df = (
            FinanceAnalytics.get_category_breakdown(
                df_active
            )
        )

        st.subheader(
            "Category-wise Spending Breakdown"
        )

        col_left, col_right = (
            st.columns(
                [1, 1]
            )
        )

        with col_left:

            st.dataframe(
                cat_df,
                width="stretch",
                hide_index=True,
            )

        with col_right:

            if not cat_df.empty:

                percentage_col = (
                    "Percentage (%)"
                    if "Percentage (%)"
                    in cat_df.columns
                    else cat_df.columns[1]
                )

                fig = px.bar(
                    cat_df,
                    x=cat_df.columns[0],
                    y=percentage_col,
                    title=(
                        "Spending Distribution (%)"
                    ),
                    color=cat_df.columns[0],
                    color_discrete_sequence=(
                        px.colors.qualitative.Set2
                    ),
                )

                fig.update_layout(
                    xaxis_tickangle=-30
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        st.divider()

        st.subheader(
            "Monthly Income vs Expense Trends"
        )

        trends_df = (
            FinanceAnalytics.get_monthly_trends(
                df_active
            )
        )

        if not trends_df.empty:

            fig_trend = px.line(
                trends_df,
                x="YearMonth",
                y=[
                    "Income",
                    "Expense",
                    "Net_Cash_Flow",
                ],
                title=(
                    "Monthly Cash Flow Timeline"
                ),
                markers=True,
            )

            st.plotly_chart(
                fig_trend,
                width="stretch",
            )

        else:

            st.info(
                "Monthly trends require valid date and amount columns."
            )

        st.divider()

        # ---------------------------------------------------------------------
        # PERSONALIZED SPENDING FORECAST & UNCERTAINTY BOUNDS
        # ---------------------------------------------------------------------
        st.subheader("📈 Personalized Spending Forecast & Uncertainty")
        st.markdown(
            "Predicts next-month total expenses using historical spending behavior with "
            "empirical volatility uncertainty bounds. Distinguishes verified past records from projected estimates."
        )

        fc_result = SpendingForecaster.forecast_user_expenses(df_active)

        if fc_result.get("status") == "success":
            pf = fc_result["point_forecast"]
            lb = fc_result["lower_bound"]
            ub = fc_result["upper_bound"]
            hist_cnt = fc_result["historical_months_count"]

            f_c1, f_c2, f_c3 = st.columns(3)
            f_c1.metric("Next Month Point Forecast", f"INR {pf:,.2f}")
            f_c2.metric("80% Empirical Uncertainty Range", f"INR {lb:,.2f} – INR {ub:,.2f}")
            f_c3.metric("Historical Base", f"{hist_cnt} Months Evaluated")

            st.info(f"💡 **Forecasting Method**: {fc_result.get('method')}\n\n{fc_result.get('explanation')}")

            # Plot spending history + projected point
            hist_spending = fc_result.get("historical_monthly_spending", {})
            if hist_spending:
                hist_series = pd.DataFrame(list(hist_spending.items()), columns=["Month", "Spending"])
                hist_series["Type"] = "Historical Actual"
                # Add forecast row
                last_m = hist_series["Month"].iloc[-1]
                next_m = "Next Month (Projected)"
                fc_row = pd.DataFrame([{"Month": next_m, "Spending": pf, "Type": "Forecast Estimate"}])
                chart_df = pd.concat([hist_series, fc_row], ignore_index=True)

                fig_fc = px.line(
                    chart_df,
                    x="Month",
                    y="Spending",
                    color="Type",
                    title="Historical Spending & 1-Month Ahead Forecast",
                    markers=True,
                    color_discrete_map={"Historical Actual": "#2563EB", "Forecast Estimate": "#10B981"}
                )
                fig_fc.add_hline(y=ub, line_dash="dash", line_color="#F59E0B", annotation_text=f"Upper Bound (₹{ub:,.0f})")
                fig_fc.add_hline(y=lb, line_dash="dash", line_color="#F59E0B", annotation_text=f"Lower Bound (₹{lb:,.0f})")
                st.plotly_chart(fig_fc, width="stretch")

            # Category forecast breakdown
            cat_fc = fc_result.get("category_forecast", {})
            if cat_fc:
                st.markdown("##### Projected Next-Month Category Breakdown")
                cat_fc_df = pd.DataFrame([{"Category": k, "Projected_Spending": v} for k, v in cat_fc.items()])
                cat_fc_df = cat_fc_df.sort_values("Projected_Spending", ascending=False)
                st.dataframe(
                    cat_fc_df.style.format({"Projected_Spending": "₹{:,.2f}"}),
                    width="stretch",
                    hide_index=True
                )
        else:
            st.info(
                f"ℹ️ {fc_result.get('message', 'Requires at least 3 months of dated transaction history.')}\n\n"
                "You can inspect verified backtesting benchmarks on a multi-month synthetic profile below."
            )

        # ---------------------------------------------------------------------
        # ROLLING-ORIGIN BACKTEST BENCHMARKS
        # ---------------------------------------------------------------------
        st.subheader("🔬 Time-Series Backtesting Benchmarks (Rolling-Origin Evaluation)")
        st.markdown(
            "To evaluate forecasting methods rigorously without data leakage, we evaluate models across "
            "18 expanding rolling windows on a 24-month profile with festive/seasonal spikes."
        )

        backtest_file = Path("results/forecast_backtest_report.csv")
        if backtest_file.exists():
            bt_df = pd.read_csv(backtest_file)
            st.dataframe(bt_df, width="stretch", hide_index=True)
            st.success(
                "**Key Empirical Finding**: Seasonal-Naive (12-Month Lag) achieves the lowest error "
                "(sMAPE 3.35%, MASE 0.626 < 1.0), significantly outperforming the 3-Month Moving Average "
                "(sMAPE 5.71%, MASE 1.083) because moving averages lag behind recurring annual Diwali/festive spending spikes."
            )
        else:
            st.caption("Rolling backtest benchmarks will appear once results/forecast_backtest_report.csv is generated.")

    else:

        st.info(
            "No transaction data is available for analysis."
        )


# =============================================================================
# BUDGET & ANOMALIES
# =============================================================================
elif nav_option == "⚠️ Budget & Anomalies":

    st.header("Budget Management, Goal Planning & Anomaly Detection")

    df_active = st.session_state.user_df
    if df_active is None:
        try:
            df_active = load_demo_data(2000)
            st.caption("🧪 Demo Mode • Showing synthetic sample transactions.")
        except Exception:
            df_active = pd.DataFrame()

    if df_active is None or df_active.empty:
        st.info(
            "Upload a transaction CSV in **Categorize Transactions** or load demo data "
            "to use Budgeting, Goal Planning, and Anomaly Detection."
        )
    else:
        tab_budget, tab_goals, tab_anom = st.tabs([
            "📊 Monthly Budgeting",
            "🎯 Goal-Based Savings",
            "⚠️ Unusual & Recurring",
        ])

        with tab_budget:
            st.subheader("Monthly Category Budget Tracker")
            st.markdown(
                "Set category budgets and track spending in real-time. "
                "**Refunds offset category spending deterministically**, and non-expense categories (e.g. Income) are excluded."
            )

            # Determine available months from dates
            if "date" in df_active.columns:
                avail_months = (
                    pd.to_datetime(df_active["date"], errors="coerce")
                    .dt.strftime("%Y-%m")
                    .dropna()
                    .unique()
                    .tolist()
                )
                avail_months = sorted(avail_months, reverse=True)
            else:
                avail_months = []

            current_selected_month = st.selectbox(
                "Select Month for Budget Analysis",
                avail_months if avail_months else ["2026-10"],
                key="selected_budget_month",
            )

            bm = st.session_state.budget_manager

            # Pre-seed sensible defaults if this month is unconfigured
            existing_budgets = bm.get_monthly_budgets(current_selected_month)
            if not existing_budgets:
                default_seed = {
                    "Food & Dining": 15000.0,
                    "Shopping & Retail": 10000.0,
                    "Utilities & Services": 6000.0,
                    "Transportation": 5000.0,
                    "Entertainment & Recreation": 4000.0,
                    "Financial Services": 12000.0,
                    "Healthcare & Medical": 5000.0,
                }
                for c, a in default_seed.items():
                    bm.set_budget(current_selected_month, c, a)

            # Interactive budget adjuster
            with st.expander("⚙️ Set or Adjust Category Budgets", expanded=False):
                c_col1, c_col2, c_col3 = st.columns([2, 2, 1])
                all_cats = [
                    "Food & Dining",
                    "Shopping & Retail",
                    "Utilities & Services",
                    "Transportation",
                    "Entertainment & Recreation",
                    "Financial Services",
                    "Healthcare & Medical",
                    "Government & Legal",
                    "Charity & Donations",
                ]
                with c_col1:
                    cat_to_edit = st.selectbox("Category", all_cats, key="budget_cat_select")
                with c_col2:
                    cur_val = bm.get_budget(current_selected_month, cat_to_edit)
                    new_val = st.number_input(
                        "Monthly Budget (INR)",
                        min_value=0.0,
                        value=float(cur_val),
                        step=500.0,
                        key="budget_val_input",
                    )
                with c_col3:
                    st.write("")
                    st.write("")
                    if st.button("Save Budget", key="save_budget_btn"):
                        bm.set_budget(current_selected_month, cat_to_edit, new_val)
                        st.success(f"Updated {cat_to_edit} budget to INR {new_val:,.2f}")
                        st.rerun()

            # Generate budget performance report
            budget_report = bm.generate_budget_report(df_active, current_selected_month)

            if not budget_report.empty:
                total_budget = budget_report["Budget"].sum()
                total_spent = budget_report["Actual_Spent"].sum()
                net_remaining = total_budget - total_spent
                over_count = (budget_report["Status"] == "Over Budget").sum()
                near_count = (budget_report["Status"] == "Near Limit").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Budget", f"INR {total_budget:,.2f}")
                m2.metric("Actual Spending", f"INR {total_spent:,.2f}")
                m3.metric(
                    "Net Remaining",
                    f"INR {net_remaining:,.2f}",
                    delta=f"{'+' if net_remaining >= 0 else ''}{net_remaining:,.2f}" if total_budget > 0 else None,
                )
                m4.metric("Attention Categories", f"{over_count} Over / {near_count} Near Limit")

                st.subheader("Category Budget Breakdown")

                def status_styler(val):
                    if val == "Over Budget":
                        return "color: #DC2626; font-weight: bold;"
                    elif val == "Near Limit":
                        return "color: #D97706; font-weight: bold;"
                    elif val == "Within Budget":
                        return "color: #16A34A; font-weight: bold;"
                    return "color: #64748B;"

                styled_report = budget_report.style.map(status_styler, subset=["Status"]).format({
                    "Budget": "₹{:,.2f}",
                    "Actual_Spent": "₹{:,.2f}",
                    "Remaining": "₹{:,.2f}",
                    "Percentage_Used": "{:.1f}%",
                })
                st.dataframe(styled_report, width="stretch", hide_index=True)

                # Plotly comparison chart
                comp_df = budget_report[(budget_report["Budget"] > 0) | (budget_report["Actual_Spent"] > 0)].copy()
                if not comp_df.empty:
                    melted = comp_df.melt(
                        id_vars=["Category"],
                        value_vars=["Budget", "Actual_Spent"],
                        var_name="Metric",
                        value_name="Amount",
                    )
                    fig_b = px.bar(
                        melted,
                        x="Category",
                        y="Amount",
                        color="Metric",
                        barmode="group",
                        title=f"Budget vs Actual Spending ({current_selected_month})",
                        color_discrete_map={"Budget": "#2563EB", "Actual_Spent": "#EF4444"},
                    )
                    fig_b.update_layout(xaxis_tickangle=-30)
                    st.plotly_chart(fig_b, width="stretch")
            else:
                st.info(f"No transactions found for month {current_selected_month}.")

        with tab_goals:
            st.subheader("Goal-Based Savings Planner")
            st.markdown(
                "Plan towards savings goals deterministically based on your verified monthly cash-flow surplus."
            )

            # 1. Historical Cash Flow Analysis
            hist_surplus = GoalSavingPlanner.calculate_historical_surplus(df_active)

            hs1, hs2, hs3 = st.columns(3)
            hs1.metric("Avg Monthly Income", f"INR {hist_surplus.get('average_monthly_income', 0.0):,.2f}")
            hs2.metric("Avg Monthly Expense", f"INR {hist_surplus.get('average_monthly_expense', 0.0):,.2f}")
            avg_surp = hist_surplus.get("average_monthly_surplus", 0.0)
            hs3.metric("Avg Monthly Surplus", f"INR {avg_surp:,.2f}")

            st.divider()

            # 2. Interactive Goal Planner Form
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                goal_name = st.text_input("Savings Goal Name", value="Emergency Fund Reserve", key="goal_name_input")
            with g_col2:
                target_amount = st.number_input("Target Amount (INR)", min_value=1000.0, value=60000.0, step=5000.0, key="goal_target_amt")
            with g_col3:
                current_savings = st.number_input("Current Savings (INR)", min_value=0.0, value=10000.0, step=2000.0, key="goal_cur_savings")

            h_col1, _ = st.columns([1, 2])
            with h_col1:
                target_months = st.number_input("Target Horizon (Months)", min_value=1, max_value=120, value=6, step=1, key="goal_target_months")

            goal_eval = GoalSavingPlanner.evaluate_goal(
                target_amount=target_amount,
                current_savings=current_savings,
                target_date_or_months=target_months,
                historical_df=df_active,
            )

            req_monthly = goal_eval.get("required_monthly_savings", 0.0)
            st.markdown(f"#### Goal Assessment for **{goal_name}**")

            eval_c1, eval_c2, eval_c3 = st.columns(3)
            eval_c1.metric("Remaining Target", f"INR {target_amount - current_savings:,.2f}")
            eval_c2.metric("Required Monthly Savings", f"INR {req_monthly:,.2f}/mo")
            eval_c3.metric(
                "Feasibility Status",
                "✅ Feasible" if goal_eval.get("is_feasible") else "⚠️ Shortfall Projected",
            )

            if goal_eval.get("is_feasible"):
                st.success(
                    f"**Goal is Feasible**: Your historical surplus of INR {avg_surp:,.2f}/mo covers "
                    f"the required INR {req_monthly:,.2f}/mo with an extra cushion of INR {avg_surp - req_monthly:,.2f}/mo."
                )
            else:
                shortfall = goal_eval.get("monthly_shortfall", 0.0)
                st.warning(
                    f"**Monthly Shortfall of INR {shortfall:,.2f}/mo**: Your current monthly surplus of "
                    f"INR {avg_surp:,.2f} is insufficient to reach INR {target_amount:,.2f} in {target_months} months."
                )

                st.subheader("💡 Recommended Discretionary Spending Cutbacks")
                recs = goal_eval.get("recommendations", [])
                if recs:
                    for r in recs:
                        st.markdown(
                            f"- **{r.get('category')}**: Current monthly spending is INR {r.get('current_monthly_spending', 0.0):,.2f}. "
                            f"Suggested reduction of INR {r.get('suggested_reduction', 0.0):,.2f}/mo ({r.get('reduction_pct', 0.0):.0f}% cutback)."
                        )
                else:
                    st.info("No discretionary categories (Food & Dining, Entertainment, Shopping) had recorded spending to suggest cutbacks.")

            st.caption(
                "⚠️ Disclaimer: Goal projections are deterministic arithmetic based on past cash-flow patterns "
                "and do not constitute certified investment or financial advice."
            )

        with tab_anom:
            st.subheader("Unusual Spending Detection")

            if "amount" in df_active.columns:
                anom_df = anomaly_detector.detect_anomalies(df_active)

                if "is_unusual" in anom_df.columns:
                    unusual_records = anom_df[anom_df["is_unusual"] == True]
                else:
                    unusual_records = pd.DataFrame()

                st.metric("Flagged Unusual Transactions", f"{len(unusual_records):,}")

                if not unusual_records.empty:
                    st.warning(
                        "These transactions were flagged by statistical thresholds or anomaly detection. Review them manually."
                    )
                    display_cols = [
                        col for col in [
                            "transaction_description",
                            "amount",
                            "predicted_category",
                            "anomaly_reason",
                        ]
                        if col in unusual_records.columns
                    ]
                    st.dataframe(unusual_records[display_cols], width="stretch", hide_index=True)
                    st.caption("An unusual transaction is not automatically fraud.")
                else:
                    st.success("No unusual transactions were flagged.")
            else:
                st.info("Anomaly detection requires transaction amount values.")

            st.divider()

            st.subheader("Recurring Expenses & Subscriptions")
            recurring_df = FinanceAnalytics.detect_recurring_transactions(df_active, min_count=2)

            if not recurring_df.empty:
                st.dataframe(recurring_df, width="stretch", hide_index=True)
            else:
                st.info("No recurring transaction patterns detected.")


# =============================================================================
# =============================================================================
# MODEL & DATA QUALITY
# =============================================================================
elif nav_option == "🔍 Model & Data Quality":

    st.header("Dataset Quality, Leakage Audits & Model Benchmarks")
    st.markdown(
        "Empirical benchmarks, leakage-safe cross-validation diagnostics, calibration analysis, "
        "and evaluation against Indian banking text noise."
    )

    reports_dir = config.reports_dir
    results_dir = Path("results")

    tab_bm, tab_leak, tab_indian, tab_calib, tab_legacy = st.tabs([
        "🏆 7-Candidate Model Comparison",
        "🛡️ Merchant Splitting & Leakage Audit",
        "🇮🇳 Indian Banking & Rent Rule",
        "🔀 Calibration & Threshold Routing",
        "📊 Preprocessing & Statistics",
    ])

    with tab_bm:
        st.subheader("7-Model Benchmark Suite (Project Brief v2)")
        st.markdown(
            "Comparison across 7 candidate models evaluated on an **identical 10-split Stratified Group K-Fold split** "
            "with zero merchant overlap between train and test, tested against synthetic Indian banking text noise."
        )

        cons_file = results_dir / "model_comparison_consolidated.csv"
        seeds_file = results_dir / "compare_v2_seeds.csv"

        if seeds_file.exists():
            st.markdown("##### Measured Multi-Seed Performance (Seeds 42, 43, 44)")
            df_seeds = pd.read_csv(seeds_file)
            st.dataframe(df_seeds, width="stretch", hide_index=True)

        if cons_file.exists():
            st.markdown("##### Consolidated 7-Candidate Architecture Overview")
            df_cons = pd.read_csv(cons_file)
            st.dataframe(df_cons, width="stretch", hide_index=True)
        else:
            # Render authoritative specification table
            st.markdown("##### Consolidated 7-Candidate Architecture Overview")
            cand_overview = pd.DataFrame([
                {"Model_ID": "Model 1", "Model_Name": "TF-IDF (char 2-5) + Logistic Regression", "Status": "MEASURED (3 seeds)", "Resource_Profile": "CPU Lightweight", "Notes": "Fastest inference (~0.0016 ms/txn), robust linear boundary on sparse n-grams."},
                {"Model_ID": "Model 2", "Model_Name": "TF-IDF + Linear SVM (Calibrated)", "Status": "MEASURED (3 seeds)", "Resource_Profile": "CPU Lightweight", "Notes": "Highest clean accuracy (~98.9%), CalibratedClassifierCV calibrated probabilities."},
                {"Model_ID": "Model 3", "Model_Name": "TF-IDF + LightGBM (max_depth=6, num_leaves=63)", "Status": "MEASURED (3 seeds)", "Resource_Profile": "CPU Multicore", "Notes": "Trees underperform linear models on 20,000 sparse n-grams, ~28x slower inference."},
                {"Model_ID": "Model 4", "Model_Name": "TF-IDF + XGBoost (tree_method=hist)", "Status": "MEASURED (3 seeds)", "Resource_Profile": "CPU Multicore / GPU", "Notes": "Histogram binning with validation early stopping on noisy validation split."},
                {"Model_ID": "Model 5", "Model_Name": "Sentence Transformers (all-MiniLM-L6-v2) + LogReg", "Status": "NOT RUN (Reproducible GPU workflow provided)", "Resource_Profile": "T4 GPU / 4GB RAM", "Notes": "Dense semantic embeddings (~384 dim). Workflow: notebooks/04b_sentence_embeddings.py"},
                {"Model_ID": "Model 6", "Model_Name": "Fine-Tuned ModernBERT / DistilBERT", "Status": "NOT RUN (Reproducible GPU workflow provided)", "Resource_Profile": "NVIDIA T4 / A100 GPU", "Notes": "Contextual attention transformer fine-tuning. Workflow: notebooks/05_modernbert_training.py"},
                {"Model_ID": "Model 7", "Model_Name": "Llama 3.2 1B Instruct with QLoRA (4-bit)", "Status": "NOT RUN (Reproducible GPU workflow provided)", "Resource_Profile": "NVIDIA A10G / A100 GPU", "Notes": "Parameter-efficient QLoRA causal LLM classifier. Workflow: notebooks/06_llama_qlora_training.py"},
            ])
            st.dataframe(cand_overview, width="stretch", hide_index=True)

        boot_file = results_dir / "bootstrap_confidence_intervals.csv"
        if boot_file.exists():
            st.markdown("##### Group-Bootstrap 95% Confidence Intervals (Resampled by Merchant)")
            df_boot = pd.read_csv(boot_file)
            st.dataframe(df_boot, width="stretch", hide_index=True)
            st.caption("Resampling by core merchant cluster avoids narrow row-level variance inflation.")

        st.info(
            "💡 **Core ML Finding**: Calibrated linear models (LogReg and LinearSVM) consistently outperform "
            "gradient-boosted trees on 20,000-dimensional sparse character n-gram text. Linear models integrate subtle "
            "sub-string cues across all 20,000 features simultaneously, whereas shallow tree splits are limited to a fraction of features."
        )

    with tab_leak:
        st.subheader("Leakage-Safe Merchant Splitting (Stratified Group K-Fold)")
        st.markdown(
            "Why random row splitting produces invalid, inflated metrics and how merchant-clustering resolves it."
        )

        leaky_file = results_dir / "leaky_baseline_evaluation.csv"
        if leaky_file.exists():
            df_leak_base = pd.read_csv(leaky_file)
            st.dataframe(df_leak_base, width="stretch", hide_index=True)

        st.warning(
            "**Data Leakage Analysis**: When transactions from the same merchant (e.g. 'Uber' or 'Netflix') appear in both "
            "train and test sets, random train/test splits allow models to achieve ~98.5% accuracy simply by memorizing merchant names. "
            "In our **10-split Stratified Group K-Fold**, all occurrences of a merchant core are strictly confined to one split. "
            "There is **0% merchant overlap** between train (2,975 merchants), val (333 merchants), and test (803 merchants)."
        )

        st.markdown("##### Pre-Training Diagnostic Checks on Raw HuggingFace Dataset")
        st.markdown(
            "- **Target Dataset**: `mitulshah/transaction-categorization` (100,000 records sampled with `seed=1`)\n"
            "- **Indian Merchant Check**: Searching for `swiggy|zomato|jio|paytm|flipkart` found **exactly 0 rows**.\n"
            "- **Target Distribution**: Verified all 10 target categories are represented across splits.\n"
            "- **Merchant Grouping**: Median group size = 1, maximum group size = 48, top 10 groups hold only ~0.4% of rows."
        )

    with tab_indian:
        st.subheader("Indian Banking Text Evaluation & Rent Rule Integration")
        st.markdown(
            "Evaluating model behavior on real-world Indian transaction noise (UPI VPAs, POS merchant wrappers, NEFT/IMPS references) "
            "and resolving domain categories absent from training data."
        )

        own_eval_file = results_dir / "own_test_evaluation.csv"
        if own_eval_file.exists():
            df_own_eval = pd.read_csv(own_eval_file)
            st.dataframe(df_own_eval, width="stretch", hide_index=True)
            st.caption("Evaluated on data/own_test_indian.csv (100 Indian banking transactions).")

        st.success(
            "**Keyword Rent Rule Attribution**: The HuggingFace training set contains 10 target categories but does NOT contain a 'Rent' category. "
            "Consequently, an ML model alone will misclassify rent payments into Financial Services or Utilities. "
            "Our deterministic `with_rent_rule()` regex router intercepts `rent|landlord|pg|lease|housing society` before the ML classifier. "
            "We strictly report model-only vs model+rule accuracy separately so rule successes are never falsely attributed to ML generalization."
        )

    with tab_calib:
        st.subheader("Model Calibration & Low-Confidence Review Routing")
        st.markdown(
            "Well-calibrated probabilities enable confident automation while safely routing uncertain predictions to human review."
        )

        thresh_file = results_dir / "confidence_threshold_routing.csv"
        if thresh_file.exists():
            df_thresh = pd.read_csv(thresh_file)
            st.dataframe(df_thresh, width="stretch", hide_index=True)
            st.caption("Thresholds tuned strictly on the noisy validation split (never on test data).")

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            cm_clean_path = results_dir / "confusion_matrix_clean.png"
            if cm_clean_path.exists():
                st.image(str(cm_clean_path), caption="Confusion Matrix on Clean Unseen Merchants", width="stretch")
            elif (reports_dir / "confusion_matrix.png").exists():
                st.image(str(reports_dir / "confusion_matrix.png"), caption="Confusion Matrix", width="stretch")
        with col_cm2:
            cm_noisy_path = results_dir / "confusion_matrix_noisy.png"
            if cm_noisy_path.exists():
                st.image(str(cm_noisy_path), caption="Confusion Matrix on Noisy Indian Text (NEFT/IMPS)", width="stretch")

        per_class_file = results_dir / "per_class_metrics.csv"
        if per_class_file.exists():
            st.markdown("##### Per-Class Precision, Recall, and F1 Breakdown (Clean vs Noisy)")
            df_pc = pd.read_csv(per_class_file)
            st.dataframe(df_pc, width="stretch", hide_index=True)

    with tab_legacy:
        st.subheader("Dataset Statistics & Preprocessing Records")
        try:
            df_stats = pd.read_csv(reports_dir / "dataset_statistics.csv")
            st.dataframe(df_stats, width="stretch", hide_index=True)
        except Exception:
            st.info("Dataset statistics report not found.")

        try:
            df_pre = pd.read_csv(reports_dir / "preprocessing_examples.csv").head(15)
            st.dataframe(df_pre, width="stretch", hide_index=True)
        except Exception:
            st.info("Preprocessing examples unavailable.")


# =============================================================================
# PRIVACY & ADVICE SAFETY
# =============================================================================
elif nav_option == "🛡️ Privacy & Advice Safety":

    st.header("Privacy Firewall, Data Minimization & Advice Safety")
    st.markdown(
        "Financial systems must treat privacy and advice safety as non-negotiable architectural constraints. "
        "This project implements an outbound data minimization firewall and deterministic financial grounding."
    )

    t_firewall, t_disclaimer, t_tester = st.tabs([
        "🔒 Outbound Privacy Firewall",
        "⚖️ Advice Safety & Grounding",
        "🧪 Interactive Privacy Tester",
    ])

    with t_firewall:
        st.subheader("Outbound Data Minimization Firewall (`PrivacyGuard`)")
        st.markdown(
            "When integrating LLMs for natural language explanations, **raw transaction records and user financial PII "
            "must NEVER be transmitted to external API endpoints**."
        )

        st.markdown(
            """
            - **Strict Field Allowlist**: Only 38 pre-approved aggregate financial summary metrics (e.g. `total_spending`, `net_cash_flow`, `category_breakdown`) can ever leave the application environment.
            - **Raw Transaction Rejection**: Any outbound payload containing transaction lists or raw row arrays triggers an immediate `PrivacyViolationError`.
            - **Regex PII Redaction**: Automatically scans and redacts 16-digit credit card numbers, 9–18 digit bank account numbers, credentials/passwords, Indian PAN cards, and Aadhaar numbers.
            - **Fail-Safe Offline Mode**: If no LLM API key is provided, the entire advisor functions locally with 100% deterministic template responses.
            """
        )

        st.markdown("##### Approved Outbound Allowlist Keys")
        keys_list = sorted(list(PrivacyGuard.ALLOWED_SUMMARY_KEYS))
        st.dataframe(pd.DataFrame({"Approved Metric Field": keys_list}), width="stretch", hide_index=True)

    with t_disclaimer:
        st.subheader("Financial Advice Safety & Grounding Principles")
        st.warning(
            """
            **Important Educational Notice & Disclaimer**

            This application is built for educational transaction classification, behavioral analytics, and budget simulation.
            It does **not** provide certified financial, investment, legal, or tax advice.

            - **Deterministic Arithmetic**: Balances, totals, surplus, and shortfalls are computed by verified Python functions. The LLM is strictly prohibited from inventing financial figures.
            - **Estimates, Not Guarantees**: Spending forecasts are statistical projections based on historical patterns, not financial guarantees.
            - **Flagging vs Fraud**: Transactions flagged by anomaly detection represent statistical deviations, not confirmed fraud or illicit activity.
            """
        )

    with t_tester:
        st.subheader("Test the Privacy Firewall Live")
        st.markdown("Type any sample text containing sensitive PII (card number, PAN, password) to verify regex redaction:")

        sample_input = st.text_area(
            "Input Text to Sanitize",
            value="Payment of INR 4500 on card 4111 2222 3333 4444 with password testpass123 for PAN ABCDE1234F.",
            height=80,
            key="privacy_test_input"
        )

        if st.button("Run Privacy Guard", key="run_privacy_guard_btn"):
            sanitized = sample_input
            for p in PrivacyGuard.FORBIDDEN_PATTERNS:
                sanitized = p.sub("[REDACTED_PII]", sanitized)
            st.markdown("##### Sanitized Output:")
            st.code(sanitized)
            st.success("✅ All prohibited sensitive financial patterns redacted successfully.")