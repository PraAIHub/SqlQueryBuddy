"""Query optimization and performance analysis"""
from typing import Dict, List, Optional, Any


class QueryOptimizer:
    """Analyzes and suggests query optimizations"""

    def __init__(self):
        self.optimization_rules = [
            self._check_missing_where_clause,
            self._check_select_star,
            self._check_missing_indexes,
            self._check_join_optimization,
            self._check_subquery_opportunity,
        ]

    def analyze(self, sql_query: str) -> Dict[str, Any]:
        """Analyze a query and provide optimization suggestions"""
        suggestions = []

        for rule in self.optimization_rules:
            suggestion = rule(sql_query)
            if suggestion:
                suggestions.append(suggestion)

        return {
            "total_suggestions": len(suggestions),
            "suggestions": suggestions,
            "optimization_level": self._calculate_optimization_level(suggestions),
        }

    @staticmethod
    def _check_missing_where_clause(query: str) -> Optional[Dict[str, str]]:
        """Check if query is missing WHERE clause"""
        query_upper = query.upper()

        if "SELECT" in query_upper and "WHERE" not in query_upper:
            if "LIMIT" not in query_upper:
                return {
                    "type": "performance",
                    "severity": "high",
                    "suggestion": "Add a WHERE clause to filter results and improve performance",
                    "example": "Add 'WHERE column = value' to limit the result set",
                }

        return None

    @staticmethod
    def _check_select_star(query: str) -> Optional[Dict[str, str]]:
        """Check for SELECT * usage"""
        if "SELECT *" in query.upper():
            return {
                "type": "efficiency",
                "severity": "medium",
                "suggestion": "Specify only needed columns instead of SELECT *",
                "example": "Use 'SELECT id, name, email' instead of 'SELECT *'",
            }

        return None

    @staticmethod
    def _check_missing_indexes(query: str) -> Optional[Dict[str, str]]:
        """Check for potential index opportunities"""
        query_upper = query.upper()

        if "WHERE" in query_upper or "ORDER BY" in query_upper:
            return {
                "type": "optimization",
                "severity": "medium",
                "suggestion": "Ensure columns in WHERE and ORDER BY clauses are indexed",
                "example": "Create indexes on frequently filtered and sorted columns",
            }

        return None

    @staticmethod
    def _check_join_optimization(query: str) -> Optional[Dict[str, str]]:
        """Check for potential JOIN optimizations"""
        query_upper = query.upper()

        if "JOIN" in query_upper:
            join_count = query_upper.count("JOIN")
            if join_count > 3:
                return {
                    "type": "performance",
                    "severity": "medium",
                    "suggestion": f"Consider refactoring {join_count} joins",
                    "example": "Complex joins can impact performance. Review if all joins are necessary",
                }

        return None

    @staticmethod
    def _check_subquery_opportunity(query: str) -> Optional[Dict[str, str]]:
        """Check for subquery opportunities"""
        if query.count("(") > 2:
            return {
                "type": "readability",
                "severity": "low",
                "suggestion": "Complex nested queries might benefit from CTEs (WITH clause)",
                "example": "Use 'WITH temp_table AS (SELECT ...) SELECT * FROM temp_table'",
            }

        return None

    @staticmethod
    def _calculate_optimization_level(suggestions: List[Dict]) -> str:
        """Calculate overall optimization level"""
        if not suggestions:
            return "excellent"

        high_severity = sum(1 for s in suggestions if s.get("severity") == "high")
        if high_severity > 0:
            return "needs_optimization"

        return "good"


