"""Insight generation and analysis of query results"""
from typing import Dict, List, Any, Optional
from langchain.chat_models import ChatOpenAI


class InsightGenerator:
    """Generates AI-driven insights from query results"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model, temperature=0.3
        )

    def generate_insights(
        self, query_results: List[Dict[str, Any]], user_query: str
    ) -> str:
        """Generate natural language insights from results"""
        if not query_results:
            return "No data available to generate insights."

        # Prepare data summary
        data_summary = self._summarize_results(query_results)

        # Build prompt
        prompt = f"""Analyze these query results and provide actionable insights.

User Question: {user_query}

Results Summary:
{data_summary}

Results Data (first 10 rows):
{str(query_results[:10])}

Provide 2-3 key insights or patterns from this data. Be specific and actionable."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "Unable to generate insights from the results."

    @staticmethod
    def _summarize_results(results: List[Dict[str, Any]]) -> str:
        """Create a summary of results"""
        if not results:
            return "No results"

        summary_lines = [f"Total records: {len(results)}", "Columns:", ""]

        if results:
            for key in results[0].keys():
                summary_lines.append(f"  - {key}")

        return "\n".join(summary_lines)


class PatternDetector:
    """Detects patterns in query results"""

    @staticmethod
    def detect_numeric_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in numeric columns"""
        patterns = {}

        if not data:
            return patterns

        numeric_columns = {}
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_columns:
                        numeric_columns[key] = []
                    numeric_columns[key].append(value)

        for col_name, values in numeric_columns.items():
            if values:
                patterns[col_name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values),
                }

        return patterns

    @staticmethod
    def detect_string_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in string columns"""
        patterns = {}

        if not data:
            return patterns

        for row in data:
            for key, value in row.items():
                if isinstance(value, str):
                    if key not in patterns:
                        patterns[key] = {"unique_values": set(), "count": 0}
                    patterns[key]["unique_values"].add(value)
                    patterns[key]["count"] += 1

        # Convert sets to lists for JSON serialization
        for key in patterns:
            patterns[key]["unique_values"] = list(patterns[key]["unique_values"])
            patterns[key]["unique_count"] = len(patterns[key]["unique_values"])

        return patterns


class TrendAnalyzer:
    """Analyzes trends in data"""

    @staticmethod
    def analyze_trends(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {}

        if not data or len(data) < 2:
            return trends

        # Look for numeric columns that might represent time series
        first_row = data[0]
        numeric_cols = [k for k, v in first_row.items() if isinstance(v, (int, float))]

        for col in numeric_cols:
            values = []
            for row in data:
                if col in row and isinstance(row[col], (int, float)):
                    values.append(row[col])

            if len(values) >= 2:
                # Simple trend: increasing or decreasing
                differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
                avg_change = sum(differences) / len(differences) if differences else 0

                trend_direction = "increasing" if avg_change > 0 else "decreasing"
                trends[col] = {
                    "direction": trend_direction,
                    "average_change": round(avg_change, 2),
                    "total_change": round(values[-1] - values[0], 2),
                }

        return trends


class ResultsAnalyzer:
    """Comprehensive results analysis"""

    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()

    def analyze(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive analysis of results"""
        return {
            "record_count": len(results),
            "numeric_patterns": self.pattern_detector.detect_numeric_patterns(results),
            "string_patterns": self.pattern_detector.detect_string_patterns(results),
            "trends": self.trend_analyzer.analyze_trends(results),
        }
