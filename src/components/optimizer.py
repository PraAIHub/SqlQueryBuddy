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


class PerformanceMetrics:
    """Tracks and analyzes query performance metrics"""

    def __init__(self):
        self.query_metrics: List[Dict[str, Any]] = []

    def record_metric(
        self,
        query: str,
        execution_time_ms: float,
        row_count: int,
        success: bool,
    ) -> None:
        """Record a query execution metric"""
        self.query_metrics.append(
            {
                "query": query,
                "execution_time_ms": execution_time_ms,
                "row_count": row_count,
                "success": success,
                "efficiency": self._calculate_efficiency(execution_time_ms, row_count),
            }
        )

    @staticmethod
    def _calculate_efficiency(execution_time_ms: float, row_count: int) -> float:
        """Calculate efficiency score (0-100, higher is better)"""
        if execution_time_ms == 0:
            return 100.0

        # Rough heuristic: ideally should be < 1ms per 100 rows
        ideal_time = row_count / 100
        if execution_time_ms <= ideal_time:
            return 100.0

        efficiency = 100 - min(100, (execution_time_ms / ideal_time) * 50)
        return max(0, efficiency)

    def get_average_execution_time(self) -> float:
        """Get average execution time across all queries"""
        if not self.query_metrics:
            return 0.0

        total_time = sum(m["execution_time_ms"] for m in self.query_metrics)
        return total_time / len(self.query_metrics)

    def get_slowest_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get slowest queries"""
        sorted_metrics = sorted(
            self.query_metrics, key=lambda x: x["execution_time_ms"], reverse=True
        )
        return sorted_metrics[:limit]

    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate efficiency report"""
        if not self.query_metrics:
            return {"average_efficiency": 0, "total_queries": 0}

        avg_efficiency = sum(
            m["efficiency"] for m in self.query_metrics
        ) / len(self.query_metrics)

        return {
            "average_efficiency": round(avg_efficiency, 2),
            "total_queries": len(self.query_metrics),
            "average_execution_time_ms": round(self.get_average_execution_time(), 2),
            "slowest_queries": self.get_slowest_queries(3),
        }
