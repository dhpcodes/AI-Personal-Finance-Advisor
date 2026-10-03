import os
import re
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from src.chatbot_tools import ChatbotFinancialTools
from src.anomaly_detection import format_currency_amount
from src.utils import logger


class GroundedFinancialAssistant:
    """
    Orchestration, intent routing, and grounded LLM interface.

    Python tools calculate financial numbers deterministically.
    The LLM explains the verified results in natural language.
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
            "LLM_MODEL", "gemini-2.5-flash"
        )
        self.timeout = int(
            timeout or os.getenv("LLM_TIMEOUT_SECONDS", "30")
        )
        self.max_tokens = int(
            max_tokens or os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000")
        )

        self.api_key = api_key or self._get_secret_key()
        self.client = None
        self._init_client()

    def _get_secret_key(self) -> Optional[str]:
        """Fetch API keys without logging credentials."""
        try:
            import streamlit as st

            if self.provider in ["gemini", "google"]:
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

        if self.provider in ["gemini", "google"]:
            return (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )

        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")

        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")

        return None

    def _init_client(self):
        """Initialize the configured LLM provider."""
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
            if self.provider in ["gemini", "google"]:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
                logger.info(
                    f"Gemini LLM client initialized with model "
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

        except Exception as e:
            logger.warning(
                f"Failed to initialize {self.provider} API client: "
                f"{e}. Falling back to offline mode."
            )
            self.client = None

    def route_intent(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Identify the user's intent and select a financial tool."""
        q_lower = question.lower().strip()

        last_intent = None
        last_category = None

        if history:
            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue

                intent = msg.get("intent")
                tool_res = msg.get("tool_result") or msg.get("details")

                if intent and not last_intent:
                    last_intent = intent

                if isinstance(tool_res, dict):
                    if not last_category:
                        last_category = (
                            tool_res.get("category")
                            or tool_res.get("highest_spending_category")
                        )

                    if not last_intent:
                        if "recurring_expenses" in tool_res:
                            last_intent = "recurring_expenses"
                        elif "category_breakdown" in tool_res:
                            last_intent = "spending_breakdown"
                        elif "monthly_cash_flow" in tool_res:
                            last_intent = "monthly_cash_flow"
                        elif "unusual_transactions" in tool_res:
                            last_intent = "detect_anomalies"
                        elif "current_spending" in tool_res:
                            last_intent = "category_period_spending"

                if last_intent and (
                    last_category
                    or last_intent == "recurring_expenses"
                ):
                    break

        # 1. Explain a transaction classification.
        if any(
            word in q_lower
            for word in ["why", "explain", "label"]
        ) and any(
            word in q_lower
            for word in [
                "classified",
                "category",
                "entertainment",
                "retail",
                "shopping",
                "food",
                "housing",
                "rent",
            ]
        ):
            match = re.search(r"['\"]([^'\"]+)['\"]", question)
            description = match.group(1) if match else question

            return {
                "intent": "explain_classification",
                "tool_name": "explain_transaction_classification",
                "args": {
                    "transaction_description": description
                },
            }

        # 2. Forecasting.
        if any(
            word in q_lower
            for word in [
                "forecast",
                "predict",
                "next month",
                "future",
                "expect",
            ]
        ):
            return {
                "intent": "forecast_expenses",
                "tool_name": "forecast_monthly_expenses",
                "args": {},
            }

        # 3. Recurring expenses.
        recurring_keywords = [
            "recurring",
            "subscription",
            "repeat",
            "monthly payment",
        ]

        is_recurring = any(
            word in q_lower for word in recurring_keywords
        )

        asks_highest_recurring = (
            is_recurring
            or last_intent == "recurring_expenses"
        ) and any(
            word in q_lower
            for word in [
                "highest",
                "most expensive",
                "largest",
                "max",
                "top",
                "greatest",
                "highest average",
            ]
        )

        if asks_highest_recurring:
            return {
                "intent": "highest_recurring_expense",
                "tool_name": "get_highest_recurring_expense",
                "args": {},
            }

        if is_recurring or "netflix" in q_lower:
            return {
                "intent": "recurring_expenses",
                "tool_name": "get_recurring_expenses",
                "args": {},
            }

        # 4. Unusual spending and anomalies.
        if any(
            word in q_lower
            for word in [
                "unusual",
                "anomaly",
                "anomalies",
                "outlier",
                "suspicious",
                "high",
            ]
        ):
            return {
                "intent": "detect_anomalies",
                "tool_name": "detect_unusual_spending",
                "args": {},
            }

        # 5. Monthly cash flow and income.
        if any(
            word in q_lower
            for word in [
                "monthly",
                "income vs expense",
                "cash flow",
                "trend",
                "salary",
            ]
        ):
            return {
                "intent": "monthly_cash_flow",
                "tool_name": "get_monthly_income_expense",
                "args": {},
            }

        # 6. Category-specific spending.
        categories = [
            "food",
            "dining",
            "shopping",
            "retail",
            "utilities",
            "transportation",
            "healthcare",
            "medical",
            "entertainment",
            "recreation",
            "government",
            "financial",
            "groceries",
            "rent",
        ]

        found_category = next(
            (
                category
                for category in categories
                if category in q_lower
            ),
            None,
        )

        if found_category:
            category_map = {
                "food": "Food & Dining",
                "dining": "Food & Dining",
                "groceries": "Food & Dining",
                "shopping": "Shopping & Retail",
                "retail": "Shopping & Retail",
                "utilities": "Utilities & Services",
                "transportation": "Transportation",
                "healthcare": "Healthcare & Medical",
                "medical": "Healthcare & Medical",
                "entertainment": "Entertainment & Recreation",
                "recreation": "Entertainment & Recreation",
                "government": "Government & Legal",
                "financial": "Financial Services",
            }

            target_category = category_map.get(
                found_category,
                found_category.capitalize(),
            )

            return {
                "intent": "category_period_spending",
                "tool_name": "get_category_spending_for_period",
                "args": {"category": target_category},
            }

        # 7. Follow-up questions.
        if last_intent and any(
            phrase in q_lower
            for phrase in [
                "last month",
                "previous month",
                "compare",
                "those transactions",
                "show details",
            ]
        ):
            if (
                "transactions" in q_lower
                or "details" in q_lower
            ):
                return {
                    "intent": "transaction_details",
                    "tool_name": "get_transaction_details",
                    "args": {"category": last_category},
                }

            if last_category:
                return {
                    "intent": "category_period_spending",
                    "tool_name": "get_category_spending_for_period",
                    "args": {"category": last_category},
                }

        # Default: spending breakdown.
        return {
            "intent": "spending_breakdown",
            "tool_name": "get_spending_by_category",
            "args": {},
        }

    def _extract_last_recurring_result(
        self,
        history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find the most recent recurring-expense result in history."""
        if not history:
            return None, None

        for msg in reversed(history):
            if not isinstance(msg, dict):
                continue

            intent = msg.get("intent")
            tool_result = msg.get("tool_result") or msg.get("details")

            if isinstance(tool_result, dict):
                if (
                    "recurring_expenses" in tool_result
                    or intent == "recurring_expenses"
                ):
                    return intent or "recurring_expenses", tool_result

        return None, None

    def process_query(
        self,
        question: str,
        df: pd.DataFrame,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run intent routing, tool execution, and response generation."""
        intent_info = self.route_intent(
            question,
            history=history,
        )

        tool_name = intent_info["tool_name"]
        args = intent_info.get("args", {})

        # Execute the financial calculation using Python tools.
        if tool_name == "get_spending_by_category":
            tool_result = (
                ChatbotFinancialTools.get_spending_by_category(df)
            )

        elif tool_name == "get_category_spending_for_period":
            tool_result = (
                ChatbotFinancialTools.get_category_spending_for_period(
                    df, **args
                )
            )

        elif tool_name == "get_recurring_expenses":
            tool_result = (
                ChatbotFinancialTools.get_recurring_expenses(df)
            )

        elif tool_name == "get_highest_recurring_expense":
            _, previous_result = self._extract_last_recurring_result(
                history
            )
            tool_result = (
                ChatbotFinancialTools.get_highest_recurring_expense(
                    previous_result=previous_result
                )
            )

        elif tool_name == "explain_transaction_classification":
            tool_result = (
                ChatbotFinancialTools.explain_transaction_classification(
                    **args
                )
            )

        elif tool_name == "forecast_monthly_expenses":
            tool_result = (
                ChatbotFinancialTools.forecast_monthly_expenses(df)
            )

        elif tool_name == "get_monthly_income_expense":
            tool_result = (
                ChatbotFinancialTools.get_monthly_income_expense(df)
            )

        elif tool_name == "detect_unusual_spending":
            tool_result = (
                ChatbotFinancialTools.detect_unusual_spending(df)
            )

        elif tool_name == "get_transaction_details":
            tool_result = (
                ChatbotFinancialTools.get_transaction_details(
                    df, **args
                )
            )

        else:
            tool_result = (
                ChatbotFinancialTools.get_spending_by_category(df)
            )

        # Generate a response with Gemini or the offline fallback.
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
            "intent": intent_info["intent"],
            "tool_used": tool_name,
            "tool_result": tool_result,
            "answer": response["answer"],
            "formatted_details": response["details"],
            "is_llm_powered": self.client is not None,
        }

    def _generate_llm_response(
        self,
        question: str,
        intent_info: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a concise, grounded response from the configured LLM."""
        system_prompt = (
            "You are a helpful AI personal finance assistant. "
            "Explain financial information in simple, natural, practical language.\n\n"
            "GROUNDING RULES:\n"
            "1. Treat the supplied Python tool output as the source of truth.\n"
            "2. Never invent amounts, transaction counts, dates, income, balances, "
            "trends, or financial facts.\n"
            "3. Use the currency provided by the tool output. Never change currencies.\n"
            "4. If information is missing, state that briefly instead of guessing.\n"
            "5. Do not call data monthly or yearly unless the data supports that period.\n"
            "6. A category named Investment does not prove investment performance, "
            "returns, or future wealth growth.\n"
            "7. Distinguish expenses, investments, and income when the verified "
            "output provides enough information. Do not silently change totals.\n"
            "8. Keep suggestions proportional to the evidence. Do not assume a category "
            "is wasteful or that the user can afford a particular budget.\n\n"
            "RESPONSE STYLE:\n"
            "- Answer the exact question first.\n"
            "- Use a friendly, conversational tone and simple words.\n"
            "- Be concise by default.\n"
            "- Do not repeat the entire breakdown unless requested or needed.\n"
            "- Use short bullet points or numbered steps when useful.\n"
            "- Include amounts and percentages only when supported by the verified output.\n"
            "- For advice questions, give practical suggestions grounded in the data.\n"
            "- For simple questions, answer directly without unnecessary headings.\n"
            "- Include methods, periods, and limitations only when relevant.\n"
            "- Never claim to have performed calculations absent from the tool output.\n"
            "- Do not present financial guidance as a guarantee or as a personalized "
            "investment recommendation.\n"
        )

        prompt = (
            f"User question:\n{question}\n\n"
            f"Selected intent: {intent_info.get('intent', 'unknown')}\n\n"
            f"Verified Python tool output:\n{tool_result}\n\n"
            "Answer the user's question using only the verified output and rules above."
        )

        try:
            if self.provider in ["gemini", "google"]:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n{prompt}",
                )
                text_out = response.text

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

        except Exception as e:
            logger.warning(
                f"LLM API call ({self.provider}) failed: {e}. "
                "Falling back to offline template."
            )

        return self._generate_template_response(
            intent_info,
            tool_result,
        )

    def _generate_template_response(
        self,
        intent_info: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a readable offline response from verified tool output."""
        status = tool_result.get("status", "error")

        if status == "error":
            answer = (
                f"⚠️ **Unable to process question:** "
                f"{tool_result.get('message', 'Invalid transaction dataset.')}"
            )
            return {"answer": answer, "details": tool_result}

        if status == "insufficient_data":
            answer = (
                f"ℹ️ **Insufficient data:** "
                f"{tool_result.get('message', 'Not enough transaction history available.')}"
            )
            return {"answer": answer, "details": tool_result}

        if status == "not_found":
            answer = (
                f"🔍 **Not found:** "
                f"{tool_result.get('message', 'No matching records found.')}"
            )
            return {"answer": answer, "details": tool_result}

        if status == "unavailable":
            answer = (
                f"❓ **Previous results unavailable:** "
                f"{tool_result.get('message', 'Please clarify your question.')}"
            )
            return {"answer": answer, "details": tool_result}

        intent = intent_info["intent"]
        currency = tool_result.get("currency", "INR")

        if intent == "spending_breakdown":
            top_category = tool_result.get(
                "highest_spending_category", "N/A"
            )
            top_amount = tool_result.get(
                "highest_spending_amount", 0.0
            )
            total_expense = tool_result.get("total_expense", 0.0)

            lines = [
                f"- **{item['category']}**: "
                f"{format_currency_amount(item['total_amount'], currency)} "
                f"({item['percentage']}%)"
                for item in tool_result.get(
                    "category_breakdown", []
                )[:5]
            ]

            answer = (
                f"Your highest spending category is **{top_category}**, "
                f"at **{format_currency_amount(top_amount, currency)}**, "
                f"out of total recorded expenses of "
                f"**{format_currency_amount(total_expense, currency)}**.\n\n"
                f"**Top categories**\n"
                + "\n".join(lines)
                + "\n\nThis covers transactions available in the dataset; "
                "it may not represent a particular month."
            )

        elif intent == "category_period_spending":
            category = tool_result.get("category", "Selected category")
            spending = tool_result.get("current_spending", 0.0)
            count = tool_result.get("transaction_count", 0)
            period = tool_result.get(
                "current_period",
                tool_result.get("period", "all available data"),
            )

            comparison = ""
            if tool_result.get("has_previous_period", False):
                previous = tool_result.get("previous_spending", 0.0)
                percentage = tool_result.get("percentage_change", 0.0)
                change = (
                    "increase" if (percentage or 0) > 0 else "decrease"
                )
                comparison = (
                    f" This is a **{abs(percentage or 0)}% {change}** "
                    f"compared with the previous period "
                    f"({format_currency_amount(previous, currency)})."
                )

            answer = (
                f"You spent **{format_currency_amount(spending, currency)}** "
                f"on **{category}** during **{period}**, across "
                f"{count} transaction(s).{comparison}"
            )

        elif intent == "recurring_expenses":
            items = tool_result.get("recurring_expenses", [])
            count = tool_result.get("recurring_expenses_count", 0)

            lines = [
                f"- **{item['merchant']}**: {item['frequency']} occurrences, "
                f"average {format_currency_amount(item['average_amount'], currency)} "
                f"({item['classification']})"
                for item in items[:5]
            ]

            summary = (
                "\n".join(lines)
                if lines
                else "No recurring expenses detected."
            )

            answer = (
                f"I found **{count} possible recurring expense pattern(s)** "
                f"in the available data.\n\n{summary}\n\n"
                "These are patterns in transaction history, not a guarantee "
                "that each payment will recur."
            )

        elif intent == "highest_recurring_expense":
            merchant = tool_result.get("merchant", "N/A")
            average = tool_result.get("average_amount", 0.0)
            frequency = tool_result.get("frequency", 0)
            classification = tool_result.get("classification", "")

            answer = (
                f"The recurring expense with the highest average amount is "
                f"**{merchant}**, averaging "
                f"**{format_currency_amount(average, currency)}** across "
                f"{frequency} occurrence(s). {classification}"
            )

        elif intent == "explain_classification":
            category = tool_result.get("predicted_category", "Unknown")
            confidence = tool_result.get("confidence_score", 0.0)
            summary = tool_result.get("explanation_summary", "")
            features = tool_result.get("important_features", [])

            feature_text = (
                ", ".join(
                    [
                        f"'{feature['feature']}' "
                        f"({feature['contribution']})"
                        for feature in features[:3]
                    ]
                )
                if features
                else "text features"
            )

            answer = (
                f"The predicted category is **{category}** with "
                f"**{confidence}% confidence**.\n\n{summary}\n\n"
                f"Key contributing terms: {feature_text}.\n\n"
                "The prediction reflects patterns learned from the model's "
                "training data."
            )

        elif intent == "forecast_expenses":
            predicted = tool_result.get("predicted_amount", 0.0)
            lower = tool_result.get("lower_bound", 0.0)
            upper = tool_result.get("upper_bound", 0.0)
            history_count = tool_result.get("historical_months_count", 0)
            recent_average = tool_result.get("recent_3_months_avg", 0.0)

            answer = (
                f"Estimated expenses for next month are "
                f"**{format_currency_amount(predicted, currency)}**, "
                f"with an estimated range of "
                f"**{format_currency_amount(lower, currency)}–"
                f"{format_currency_amount(upper, currency)}**.\n\n"
                f"This estimate uses {history_count} historical month(s). "
                f"The recent three-month average was "
                f"{format_currency_amount(recent_average, currency)}. "
                "Unexpected events can change actual spending, so treat "
                "this as an estimate rather than a guarantee."
            )

        elif intent == "detect_anomalies":
            items = tool_result.get("unusual_transactions", [])
            count = tool_result.get("unusual_transactions_count", 0)

            lines = [
                f"- **{item['description']}**: "
                f"{format_currency_amount(item['amount'], currency)} "
                f"({item['category']}) — {item['reason']}"
                for item in items[:5]
            ]

            summary = (
                "\n".join(lines)
                if lines
                else "No unusual transactions were detected."
            )

            answer = (
                f"The analysis flagged **{count} unusual transaction(s)** "
                f"in the available data.\n\n{summary}\n\n"
                "These are statistical outliers relative to the data; "
                "a flag does not by itself mean a transaction is fraudulent."
            )

        elif intent == "transaction_details":
            transactions = tool_result.get("transactions", [])
            count = tool_result.get("matched_count", 0)

            lines = [
                f"- **{transaction.get('date', 'N/A')}** | "
                f"{transaction.get('transaction_description')} | "
                f"{format_currency_amount(transaction.get('amount', 0.0), currency)} "
                f"({transaction.get('category')})"
                for transaction in transactions[:5]
            ]

            answer = (
                f"Found **{count} matching transaction(s)**.\n\n"
                + (
                    "\n".join(lines)
                    if lines
                    else "No transaction details to display."
                )
            )

        else:
            answer = (
                "The financial analysis completed, but this intent does not "
                "have a dedicated offline response template yet.\n\n"
                f"Verified result: {tool_result}"
            )

        return {
            "answer": answer,
            "details": tool_result,
        }