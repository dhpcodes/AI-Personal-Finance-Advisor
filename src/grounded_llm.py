import os
import re
from typing import Dict, Any, Optional, List

import pandas as pd

from src.chatbot_tools import ChatbotFinancialTools
from src.anomaly_detection import format_currency_amount
from src.utils import logger


class GroundedFinancialAssistant:
    """
    Grounded AI Personal Finance Assistant.

    Financial questions:
        Python tools calculate the facts first.
        Gemini explains those verified facts.

    General questions:
        Gemini answers normally.

    If Gemini is unavailable:
        deterministic local responses are used for financial questions.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.provider = (
            provider or os.getenv("LLM_PROVIDER", "gemini")
        ).lower().strip()

        self.model_name = model_name or os.getenv(
            "LLM_MODEL",
            "gemini-3.8-flash",
        )

        self.timeout = int(
            timeout or os.getenv("LLM_TIMEOUT_SECONDS", "30")
        )

        self.max_tokens = int(
            max_tokens or os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000")
        )

        self.api_key = api_key or self._get_secret_key()

        if isinstance(self.api_key, str):
            self.api_key = self.api_key.strip()

        self.client = None
        self.last_llm_error = None

        self._init_client()

    # ------------------------------------------------------------------
    # API KEY / CLIENT
    # ------------------------------------------------------------------

    def _get_secret_key(self) -> Optional[str]:
        """Read the configured API key without logging the secret."""
        try:
            import streamlit as st

            if self.provider in ("gemini", "google"):
                if "GEMINI_API_KEY" in st.secrets:
                    return str(st.secrets["GEMINI_API_KEY"])

            elif self.provider == "openai":
                if "OPENAI_API_KEY" in st.secrets:
                    return str(st.secrets["OPENAI_API_KEY"])

            elif self.provider == "anthropic":
                if "ANTHROPIC_API_KEY" in st.secrets:
                    return str(st.secrets["ANTHROPIC_API_KEY"])

        except Exception:
            pass

        if self.provider in ("gemini", "google"):
            return (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )

        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")

        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")

        return None

    def _init_client(self) -> None:
        """Initialize the selected LLM client."""
        if (
            not self.api_key
            or not self.api_key.strip()
            or self.api_key == "your_api_key_here"
        ):
            logger.info(
                f"No API key provided for {self.provider}. "
                "Running in deterministic offline mode."
            )
            return

        try:
            if self.provider in ("gemini", "google"):
                from google import genai

                self.client = genai.Client(
                    api_key=self.api_key
                )

                logger.info(
                    f"Gemini client initialized with model "
                    f"'{self.model_name}'."
                )

            elif self.provider == "openai":
                import openai

                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )

                logger.info(
                    f"OpenAI client initialized with model "
                    f"'{self.model_name}'."
                )

            elif self.provider == "anthropic":
                import anthropic

                self.client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )

                logger.info(
                    f"Anthropic client initialized with model "
                    f"'{self.model_name}'."
                )

            else:
                logger.warning(
                    f"Unsupported LLM provider '{self.provider}'. "
                    "Running in offline mode."
                )

        except Exception as exc:
            logger.warning(
                f"Failed to initialize {self.provider} client: "
                f"{exc}. Falling back to offline mode."
            )
            self.client = None

    # ------------------------------------------------------------------
    # INTENT ROUTING
    # ------------------------------------------------------------------

    def route_intent(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Route a question to either a financial tool or general chat.

        General/unmatched questions do NOT default to spending analysis.
        """

        q = (question or "").lower().strip()

        # --------------------------------------------------------------
        # 1. SPENDING BREAKDOWN
        # --------------------------------------------------------------
        spending_phrases = [
            "highest spending",
            "highest expense",
            "highest spending category",
            "top spending",
            "top spending category",
            "most spending",
            "spend the most",
            "where do i spend the most",
            "where am i spending the most",
            "what do i spend the most on",
            "what am i spending the most on",
            "largest expense",
            "biggest expense",
            "most expensive category",
            "spending breakdown",
            "breakdown of my spending",
            "breakdown of spending",
            "top expense",
            "top expenses",
            "largest spending category",
            "biggest spending category",
        ]

        if any(phrase in q for phrase in spending_phrases):
            return {
                "intent": "spending_breakdown",
                "tool_name": "get_spending_by_category",
                "args": {},
            }

        # --------------------------------------------------------------
        # 2. TRANSACTION CLASSIFICATION EXPLANATION
        # --------------------------------------------------------------
        explanation_words = [
            "why",
            "explain",
            "explanation",
            "label",
            "classified",
            "classification",
        ]

        classification_words = [
            "transaction",
            "category",
            "classified",
            "classification",
            "entertainment",
            "retail",
            "shopping",
            "food",
            "housing",
            "rent",
        ]

        if (
            any(word in q for word in explanation_words)
            and any(word in q for word in classification_words)
        ):
            match = re.search(
                r"['\"]([^'\"]+)['\"]",
                question,
            )

            description = (
                match.group(1)
                if match
                else question
            )

            return {
                "intent": "explain_classification",
                "tool_name": "explain_transaction_classification",
                "args": {
                    "transaction_description": description
                },
            }

        # --------------------------------------------------------------
        # 3. FORECASTING
        # --------------------------------------------------------------
        forecast_words = [
            "forecast",
            "predict my expenses",
            "predict expenses",
            "next month",
            "future spending",
            "future expenses",
            "expected expenses",
            "expect to spend",
        ]

        if any(word in q for word in forecast_words):
            return {
                "intent": "forecast_expenses",
                "tool_name": "forecast_monthly_expenses",
                "args": {},
            }

        # --------------------------------------------------------------
        # 4. RECURRING EXPENSES
        # --------------------------------------------------------------
        recurring_words = [
            "recurring",
            "subscription",
            "subscriptions",
            "repeat payment",
            "repeated payment",
            "monthly payment",
        ]

        if any(word in q for word in recurring_words):
            return {
                "intent": "recurring_expenses",
                "tool_name": "get_recurring_expenses",
                "args": {},
            }

        # --------------------------------------------------------------
        # 5. ANOMALIES
        # --------------------------------------------------------------
        anomaly_words = [
            "unusual spending",
            "unusual transaction",
            "unusual transactions",
            "anomaly",
            "anomalies",
            "outlier",
            "outliers",
            "suspicious transaction",
            "suspicious transactions",
        ]

        if any(word in q for word in anomaly_words):
            return {
                "intent": "detect_anomalies",
                "tool_name": "detect_unusual_spending",
                "args": {},
            }

        # --------------------------------------------------------------
        # 6. MONTHLY CASH FLOW / INCOME VS EXPENSE
        # --------------------------------------------------------------
        cash_flow_words = [
            "cash flow",
            "income vs expense",
            "income versus expense",
            "income and expense",
            "monthly income",
            "monthly expenses",
            "monthly spending",
            "monthly trend",
            "spending trend",
            "expense trend",
            "salary",
        ]

        if any(word in q for word in cash_flow_words):
            return {
                "intent": "monthly_cash_flow",
                "tool_name": "get_monthly_income_expense",
                "args": {},
            }

        # --------------------------------------------------------------
        # 7. TRANSACTION DETAILS
        # --------------------------------------------------------------
        details_words = [
            "show transactions",
            "show transaction",
            "transaction details",
            "transactions details",
            "list transactions",
            "show my transactions",
        ]

        if any(word in q for word in details_words):
            return {
                "intent": "transaction_details",
                "tool_name": "get_transaction_details",
                "args": {},
            }

        # --------------------------------------------------------------
        # 8. CATEGORY-SPECIFIC SPENDING
        # --------------------------------------------------------------
        category_map = {
            "food": "Food & Dining",
            "dining": "Food & Dining",
            "groceries": "Food & Dining",
            "shopping": "Shopping & Retail",
            "retail": "Shopping & Retail",
            "utilities": "Utilities & Services",
            "transportation": "Transportation",
            "transport": "Transportation",
            "healthcare": "Healthcare & Medical",
            "health": "Healthcare & Medical",
            "medical": "Healthcare & Medical",
            "entertainment": "Entertainment & Recreation",
            "recreation": "Entertainment & Recreation",
            "government": "Government & Legal",
            "legal": "Government & Legal",
            "financial services": "Financial Services",
            "financial": "Financial Services",
        }

        if "rent" in q and not any(
            phrase in q for phrase in spending_phrases
        ):
            return {
                "intent": "category_period_spending",
                "tool_name": "get_category_spending_for_period",
                "args": {
                    "category": "Rent"
                },
            }

        found_category = next(
            (
                keyword
                for keyword in category_map
                if keyword in q
            ),
            None,
        )

        if found_category:
            return {
                "intent": "category_period_spending",
                "tool_name": "get_category_spending_for_period",
                "args": {
                    "category": category_map[found_category]
                },
            }

        # --------------------------------------------------------------
        # 9. FOLLOW-UP QUESTIONS
        # --------------------------------------------------------------
        follow_up_words = [
            "compare",
            "previous month",
            "last month",
            "those transactions",
            "show details",
            "more details",
        ]

        if history and any(
            word in q for word in follow_up_words
        ):
            last_result = self._get_last_tool_result(history)

            if isinstance(last_result, dict):
                last_category = (
                    last_result.get("category")
                    or last_result.get(
                        "highest_spending_category"
                    )
                )

                if "transaction" in q or "details" in q:
                    return {
                        "intent": "transaction_details",
                        "tool_name": "get_transaction_details",
                        "args": {
                            "category": last_category
                        },
                    }

                if last_category:
                    return {
                        "intent": "category_period_spending",
                        "tool_name": "get_category_spending_for_period",
                        "args": {
                            "category": last_category
                        },
                    }

        # --------------------------------------------------------------
        # 10. GENERAL CHAT
        # --------------------------------------------------------------
        return {
            "intent": "general_chat",
            "tool_name": None,
            "args": {},
        }

    # ------------------------------------------------------------------
    # HISTORY HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_last_tool_result(
        history: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        if not history:
            return None

        for message in reversed(history):
            if not isinstance(message, dict):
                continue

            result = (
                message.get("tool_result")
                or message.get("details")
            )

            if isinstance(result, dict):
                return result

        return None

    # ------------------------------------------------------------------
    # TOOL EXECUTION
    # ------------------------------------------------------------------

    def process_query(
        self,
        question: str,
        df: pd.DataFrame,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        intent_info = self.route_intent(
            question,
            history=history,
        )

        intent = intent_info["intent"]
        tool_name = intent_info.get("tool_name")
        args = intent_info.get("args", {})

        # --------------------------------------------------------------
        # GENERAL CHAT
        # --------------------------------------------------------------
        if intent == "general_chat":
            response = self._generate_general_chat_response(
                question
            )

            return {
                "question": question,
                "intent": intent,
                "tool_used": None,
                "tool_result": {},
                "answer": response,
                "formatted_details": {},
                "is_llm_powered": self.client is not None,
            }

        # --------------------------------------------------------------
        # FINANCIAL TOOLS
        # --------------------------------------------------------------
        try:
            if tool_name == "get_spending_by_category":
                tool_result = (
                    ChatbotFinancialTools
                    .get_spending_by_category(df)
                )

            elif tool_name == "get_category_spending_for_period":
                tool_result = (
                    ChatbotFinancialTools
                    .get_category_spending_for_period(
                        df,
                        **args,
                    )
                )

            elif tool_name == "get_recurring_expenses":
                tool_result = (
                    ChatbotFinancialTools
                    .get_recurring_expenses(df)
                )

            elif tool_name == "explain_transaction_classification":
                tool_result = (
                    ChatbotFinancialTools
                    .explain_transaction_classification(
                        **args
                    )
                )

            elif tool_name == "forecast_monthly_expenses":
                tool_result = (
                    ChatbotFinancialTools
                    .forecast_monthly_expenses(df)
                )

            elif tool_name == "get_monthly_income_expense":
                tool_result = (
                    ChatbotFinancialTools
                    .get_monthly_income_expense(df)
                )

            elif tool_name == "detect_unusual_spending":
                tool_result = (
                    ChatbotFinancialTools
                    .detect_unusual_spending(df)
                )

            elif tool_name == "get_transaction_details":
                tool_result = (
                    ChatbotFinancialTools
                    .get_transaction_details(
                        df,
                        **args,
                    )
                )

            else:
                tool_result = {
                    "status": "error",
                    "message": (
                        "No financial tool was selected "
                        "for this question."
                    ),
                }

        except Exception as exc:
            logger.warning(
                f"Financial tool '{tool_name}' failed: {exc}"
            )

            tool_result = {
                "status": "error",
                "message": (
                    "Financial analysis could not be completed: "
                    f"{exc}"
                ),
            }

        # --------------------------------------------------------------
        # RESPONSE GENERATION
        # --------------------------------------------------------------
        if self.client:
            response = self._generate_llm_response(
                question,
                intent_info,
                tool_result,
            )
        else:
            response = self._generate_template_response(
                intent_info,
                tool_result,
            )

        return {
            "question": question,
            "intent": intent,
            "tool_used": tool_name,
            "tool_result": tool_result,
            "answer": response["answer"],
            "formatted_details": response["details"],
            "is_llm_powered": self.client is not None,
        }

    # ------------------------------------------------------------------
    # GENERAL GEMINI CHAT
    # ------------------------------------------------------------------

    def _generate_general_chat_response(
        self,
        question: str,
    ) -> str:

        if not self.client:
            return (
                "I'm your AI Personal Finance Assistant. "
                "I can answer questions about your transaction data, "
                "spending, recurring expenses, cash flow, anomalies, "
                "forecasts, and transaction categories. "
                "Please connect a valid Gemini API key for general "
                "AI conversation."
            )

        system_prompt = (
            "You are the AI assistant inside an AI Personal Finance "
            "Advisor application. "
            "Answer the user's question naturally and concisely. "
            "Do not invent personal financial data. "
            "If the user asks a financial-data question, say that "
            "financial calculations should be handled by the "
            "application's verified transaction-analysis tools."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"User question:\n{question}\n\n"
            "Answer naturally."
        )

        try:
            if self.provider in ("gemini", "google"):

                # Current Gemini Interactions API
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input=prompt,
                )

                text_out = interaction.output_text

            elif self.provider == "openai":

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )

                text_out = response.choices[0].message.content

            elif self.provider == "anthropic":

                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    max_tokens=self.max_tokens,
                )

                text_out = (
                    response.content[0].text
                    if response.content
                    else None
                )

            else:
                text_out = None

            if text_out and text_out.strip():
                return text_out.strip()

        except Exception as exc:
            self.last_llm_error = str(exc)

            safe_error = str(exc).replace(
                self.api_key or "",
                "[API_KEY_HIDDEN]",
            )

            logger.warning(
                f"General LLM request failed: "
                f"{safe_error}"
            )

            return (
                "I couldn't reach the AI service right now.\n\n"
                f"**Gemini diagnostic:** "
                f"`{type(exc).__name__}: {safe_error}`\n\n"
                f"**Model:** `{self.model_name}`\n\n"
                "The API key is being read, but the Gemini "
                "request was rejected or failed. "
                "Check the diagnostic above."
            )

        return (
            "I couldn't reach the AI service right now. "
            "No response was returned by the configured "
            "AI provider."
        )

    # ------------------------------------------------------------------
    # GROUNDED GEMINI RESPONSE
    # ------------------------------------------------------------------

    def _generate_llm_response(
        self,
        question: str,
        intent_info: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        system_prompt = (
            "You are a grounded AI personal finance assistant.\n\n"
            "GROUNDING RULES:\n"
            "1. The Python tool output is the source of truth.\n"
            "2. Never invent amounts, counts, dates, categories, "
            "income, expenses, balances, percentages, or trends.\n"
            "3. Do not change the currency.\n"
            "4. Do not silently recalculate a different result.\n"
            "5. If the tool output says information is unavailable, "
            "say so instead of guessing.\n"
            "6. Answer the exact question first.\n"
            "7. Keep the answer concise and easy to understand.\n"
            "8. Financial suggestions must be proportional to the "
            "verified data and must not be presented as guarantees.\n"
            "9. Do not claim that a transaction is fraudulent merely "
            "because it was flagged as unusual.\n"
        )

        prompt = (
            f"User question:\n{question}\n\n"
            f"Selected intent:\n"
            f"{intent_info.get('intent')}\n\n"
            f"Verified Python tool output:\n"
            f"{tool_result}\n\n"
            "Explain the verified result in natural language."
        )

        try:
            if self.provider in ("gemini", "google"):

                # Current Gemini Interactions API
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input=(
                        f"{system_prompt}\n\n"
                        f"{prompt}"
                    ),
                )

                text_out = interaction.output_text

            elif self.provider == "openai":

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )

                text_out = response.choices[0].message.content

            elif self.provider == "anthropic":

                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=self.max_tokens,
                )

                text_out = (
                    response.content[0].text
                    if response.content
                    else None
                )

            else:
                text_out = None

            if text_out and text_out.strip():
                return {
                    "answer": text_out.strip(),
                    "details": tool_result,
                }

        except Exception as exc:
            self.last_llm_error = str(exc)

            safe_error = str(exc).replace(
                self.api_key or "",
                "[API_KEY_HIDDEN]",
            )

            logger.warning(
                f"LLM API call ({self.provider}) failed: "
                f"{safe_error}. "
                "Using offline response."
            )

        return self._generate_template_response(
            intent_info,
            tool_result,
        )

    # ------------------------------------------------------------------
    # OFFLINE / DETERMINISTIC RESPONSES
    # ------------------------------------------------------------------

    def _generate_template_response(
        self,
        intent_info: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        status = tool_result.get(
            "status",
            "success",
        )

        if status == "error":
            return {
                "answer": (
                    "⚠️ **Unable to process the question:** "
                    f"{tool_result.get('message', 'Unknown error.')}"
                ),
                "details": tool_result,
            }

        if status == "insufficient_data":
            return {
                "answer": (
                    "ℹ️ **Insufficient data:** "
                    f"{tool_result.get('message', 'Not enough data available.')}"
                ),
                "details": tool_result,
            }

        if status == "not_found":
            return {
                "answer": (
                    "🔍 **Not found:** "
                    f"{tool_result.get('message', 'No matching records found.')}"
                ),
                "details": tool_result,
            }

        intent = intent_info.get("intent")
        currency = tool_result.get(
            "currency",
            "INR",
        )

        # --------------------------------------------------------------
        # SPENDING BREAKDOWN
        # --------------------------------------------------------------
        if intent == "spending_breakdown":

            top_category = tool_result.get(
                "highest_spending_category",
                "N/A",
            )

            top_amount = tool_result.get(
                "highest_spending_amount",
                0.0,
            )

            total_expense = tool_result.get(
                "total_expense",
                0.0,
            )

            breakdown = tool_result.get(
                "category_breakdown",
                [],
            )

            lines = []

            for item in breakdown[:5]:
                category = item.get(
                    "category",
                    "Unknown",
                )

                amount = item.get(
                    "total_amount",
                    0.0,
                )

                percentage = item.get(
                    "percentage",
                    0,
                )

                lines.append(
                    f"- **{category}**: "
                    f"{format_currency_amount(amount, currency)} "
                    f"({percentage}%)"
                )

            top_lines = (
                "\n".join(lines)
                if lines
                else "No category breakdown is available."
            )

            answer = (
                f"Your highest spending category is "
                f"**{top_category}**, at "
                f"**{format_currency_amount(top_amount, currency)}**, "
                f"out of total recorded expenses of "
                f"**{format_currency_amount(total_expense, currency)}**."
                f"\n\n"
                f"**Top spending categories**\n"
                f"{top_lines}"
            )

            return {
                "answer": answer,
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # CATEGORY SPENDING
        # --------------------------------------------------------------
        if intent == "category_period_spending":

            category = tool_result.get(
                "category",
                "Selected category",
            )

            spending = tool_result.get(
                "current_spending",
                0.0,
            )

            count = tool_result.get(
                "transaction_count",
                0,
            )

            period = tool_result.get(
                "current_period",
                tool_result.get(
                    "period",
                    "all available data",
                ),
            )

            answer = (
                f"You spent "
                f"**{format_currency_amount(spending, currency)}** "
                f"on **{category}** during **{period}**, "
                f"across **{count} transaction(s)**."
            )

            if tool_result.get("has_previous_period"):

                previous = tool_result.get(
                    "previous_spending",
                    0.0,
                )

                percentage = tool_result.get(
                    "percentage_change",
                    0.0,
                )

                direction = (
                    "increase"
                    if percentage > 0
                    else "decrease"
                )

                answer += (
                    f" This is a **{abs(percentage)}% "
                    f"{direction}** compared with the "
                    f"previous period "
                    f"({format_currency_amount(previous, currency)})."
                )

            return {
                "answer": answer,
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # RECURRING EXPENSES
        # --------------------------------------------------------------
        if intent == "recurring_expenses":

            items = tool_result.get(
                "recurring_expenses",
                [],
            )

            count = tool_result.get(
                "recurring_expenses_count",
                len(items),
            )

            lines = []

            for item in items[:5]:

                merchant = item.get(
                    "merchant",
                    "Unknown",
                )

                frequency = item.get(
                    "frequency",
                    0,
                )

                average = item.get(
                    "average_amount",
                    0.0,
                )

                classification = item.get(
                    "classification",
                    "",
                )

                classification_text = (
                    f" ({classification})"
                    if classification
                    else ""
                )

                lines.append(
                    f"- **{merchant}**: "
                    f"{frequency} occurrences, average "
                    f"{format_currency_amount(average, currency)}"
                    f"{classification_text}"
                )

            details = (
                "\n".join(lines)
                if lines
                else "No recurring expense patterns detected."
            )

            return {
                "answer": (
                    f"I found **{count} possible recurring "
                    f"expense pattern(s)** in the available data."
                    f"\n\n{details}"
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------------------
        if intent == "explain_classification":

            category = tool_result.get(
                "predicted_category",
                "Unknown",
            )

            confidence = tool_result.get(
                "confidence_score",
                0.0,
            )

            summary = tool_result.get(
                "explanation_summary",
                "",
            )

            features = tool_result.get(
                "important_features",
                [],
            )

            feature_parts = []

            for feature in features[:3]:

                feature_parts.append(
                    f"'{feature.get('feature', '')}' "
                    f"({feature.get('contribution', '')})"
                )

            feature_text = (
                ", ".join(feature_parts)
                if feature_parts
                else "text features"
            )

            return {
                "answer": (
                    f"The predicted category is **{category}** "
                    f"with **{confidence}% confidence**."
                    f"\n\n"
                    f"{summary}"
                    f"\n\n"
                    f"Key contributing terms: "
                    f"{feature_text}."
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # FORECAST
        # --------------------------------------------------------------
        if intent == "forecast_expenses":

            predicted = tool_result.get(
                "predicted_amount",
                0.0,
            )

            lower = tool_result.get(
                "lower_bound",
                0.0,
            )

            upper = tool_result.get(
                "upper_bound",
                0.0,
            )

            history_count = tool_result.get(
                "historical_months_count",
                0,
            )

            recent_average = tool_result.get(
                "recent_3_months_avg",
                0.0,
            )

            return {
                "answer": (
                    f"Estimated expenses for next month are "
                    f"**{format_currency_amount(predicted, currency)}**, "
                    f"with an estimated range of "
                    f"**{format_currency_amount(lower, currency)}–"
                    f"{format_currency_amount(upper, currency)}**."
                    f"\n\n"
                    f"This estimate uses "
                    f"{history_count} historical month(s). "
                    f"The recent three-month average was "
                    f"{format_currency_amount(recent_average, currency)}."
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # ANOMALIES
        # --------------------------------------------------------------
        if intent == "detect_anomalies":

            items = tool_result.get(
                "unusual_transactions",
                [],
            )

            count = tool_result.get(
                "unusual_transactions_count",
                len(items),
            )

            lines = []

            for item in items[:5]:

                description = item.get(
                    "description",
                    "Unknown transaction",
                )

                amount = item.get(
                    "amount",
                    0.0,
                )

                category = item.get(
                    "category",
                    "Unknown",
                )

                reason = item.get(
                    "reason",
                    "",
                )

                reason_text = (
                    f" — {reason}"
                    if reason
                    else ""
                )

                lines.append(
                    f"- **{description}**: "
                    f"{format_currency_amount(amount, currency)} "
                    f"({category})"
                    f"{reason_text}"
                )

            details = (
                "\n".join(lines)
                if lines
                else "No unusual transactions were detected."
            )

            return {
                "answer": (
                    f"The analysis flagged **{count} unusual "
                    f"transaction(s)** in the available data."
                    f"\n\n"
                    f"{details}"
                    f"\n\n"
                    "An unusual-spending flag is a statistical "
                    "signal; it does not by itself mean a "
                    "transaction is fraudulent."
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # MONTHLY CASH FLOW
        # --------------------------------------------------------------
        if intent == "monthly_cash_flow":

            monthly = tool_result.get(
                "monthly_cash_flow",
                [],
            )

            if isinstance(monthly, list) and monthly:

                lines = []

                for row in monthly[-5:]:

                    month = row.get(
                        "month",
                        "Unknown",
                    )

                    income = row.get(
                        "income",
                        0.0,
                    )

                    expense = row.get(
                        "expense",
                        0.0,
                    )

                    net = row.get(
                        "net_cash_flow",
                        0.0,
                    )

                    lines.append(
                        f"- **{month}** — "
                        f"Income: "
                        f"{format_currency_amount(income, currency)}, "
                        f"Expenses: "
                        f"{format_currency_amount(expense, currency)}, "
                        f"Net: "
                        f"{format_currency_amount(net, currency)}"
                    )

                return {
                    "answer": (
                        "**Monthly cash-flow summary**"
                        f"\n\n"
                        + "\n".join(lines)
                    ),
                    "details": tool_result,
                }

            return {
                "answer": (
                    "The monthly cash-flow tool completed, "
                    "but there is no monthly breakdown "
                    "available to display."
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # TRANSACTION DETAILS
        # --------------------------------------------------------------
        if intent == "transaction_details":

            transactions = tool_result.get(
                "transactions",
                [],
            )

            count = tool_result.get(
                "matched_count",
                len(transactions),
            )

            lines = []

            for transaction in transactions[:10]:

                date = transaction.get(
                    "date",
                    "N/A",
                )

                description = transaction.get(
                    "transaction_description",
                    "Unknown",
                )

                amount = transaction.get(
                    "amount",
                    0.0,
                )

                category = transaction.get(
                    "category",
                    "Unknown",
                )

                lines.append(
                    f"- **{date}** | "
                    f"{description} | "
                    f"{format_currency_amount(amount, currency)} "
                    f"({category})"
                )

            details = (
                "\n".join(lines)
                if lines
                else "No matching transaction details were found."
            )

            return {
                "answer": (
                    f"Found **{count} matching transaction(s)**."
                    f"\n\n"
                    f"{details}"
                ),
                "details": tool_result,
            }

        # --------------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------------
        return {
            "answer": (
                "The financial analysis completed, but there "
                "is no dedicated response template for this "
                "result yet."
            ),
            "details": tool_result,
        }


# ----------------------------------------------------------------------
# OPTIONAL SIMPLE SELF-TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":

    assistant = GroundedFinancialAssistant()

    test_questions = [
        "Are you working?",
        "What is my highest spending category?",
        "Where do I spend the most?",
        "What are my top spending categories?",
        "How much do I spend on food?",
        "Show my recurring expenses",
        "What are my unusual transactions?",
        "What will my expenses be next month?",
    ]

    for test_question in test_questions:
        print(
            test_question,
            "->",
            assistant.route_intent(test_question),
        )