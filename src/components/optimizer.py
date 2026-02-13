"""Query optimization and performance analysis"""
import re
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
            self._check_group_by_without_index,
            self._check_unbounded_result,
            self._check_function_on_indexed_column,
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
    def _extract_columns(clause_text: str) -> List[str]:
        """Extract column names from a SQL clause."""
        # Match word.word (table.column) or standalone words, skip SQL keywords
        skip = {
            "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "BETWEEN",
            "ASC", "DESC", "AS", "ON", "BY", "FROM", "JOIN", "WHERE",
            "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "SELECT",
        }
        # SQL functions that should not be treated as column names
        functions = {
            "STRFTIME", "DATE", "DATETIME", "TIME", "JULIANDAY",
            "LOWER", "UPPER", "SUBSTR", "SUBSTRING", "REPLACE", "TRIM",
            "CAST", "COALESCE", "IFNULL", "NULLIF", "ABS", "ROUND",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "LENGTH", "TYPEOF",
            "NOW", "TOTAL", "GROUP_CONCAT",
        }
        tokens = re.findall(r"(?:\w+\.)?(\w+)", clause_text)
        return [
            t for t in tokens
            if t.upper() not in skip
            and t.upper() not in functions
            and not t.isdigit()
            and len(t) > 1  # skip single-char aliases and format specifiers
        ]

    @staticmethod
    def _check_missing_where_clause(query: str) -> Optional[Dict[str, str]]:
        """Check if query is missing WHERE clause"""
        query_upper = query.upper()

        if "SELECT" in query_upper and "WHERE" not in query_upper:
            if "LIMIT" not in query_upper:
                return {
                    "type": "performance",
                    "severity": "high",
                    "suggestion": "Add a WHERE clause or LIMIT to avoid scanning the entire table",
                    "example": "Add 'WHERE column = value' or 'LIMIT 100' to bound the result set",
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
        """Suggest specific indexes based on WHERE and ORDER BY columns"""
        query_upper = query.upper()
        index_cols = []

        # Extract WHERE columns
        where_match = re.search(r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)", query_upper, re.DOTALL)
        if where_match:
            index_cols.extend(QueryOptimizer._extract_columns(where_match.group(1)))

        # Extract ORDER BY columns
        order_match = re.search(r"ORDER\s+BY\s+(.+?)(?:LIMIT|OFFSET|$)", query_upper, re.DOTALL)
        if order_match:
            index_cols.extend(QueryOptimizer._extract_columns(order_match.group(1)))

        if index_cols:
            unique_cols = list(dict.fromkeys(c.lower() for c in index_cols))[:4]
            col_list = ", ".join(unique_cols)
            return {
                "type": "optimization",
                "severity": "medium",
                "suggestion": f"Consider adding indexes on: {col_list}",
                "example": f"CREATE INDEX idx_{unique_cols[0]} ON table({unique_cols[0]})",
            }

        return None

    @staticmethod
    def _check_join_optimization(query: str) -> Optional[Dict[str, str]]:
        """Check for potential JOIN optimizations with specific advice"""
        query_upper = query.upper()

        if "JOIN" in query_upper:
            join_count = query_upper.count("JOIN")
            # Extract join columns
            on_matches = re.findall(r"ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", query, re.IGNORECASE)
            if join_count >= 2 and on_matches:
                join_cols = [f"{m[0]}={m[1]}" for m in on_matches[:3]]
                return {
                    "type": "performance",
                    "severity": "medium" if join_count <= 3 else "high",
                    "suggestion": f"Ensure join columns are indexed ({', '.join(join_cols)})",
                    "example": "Index foreign key columns used in ON clauses for faster joins",
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
                "example": "Use 'WITH temp AS (SELECT ...) SELECT * FROM temp'",
            }

        return None

    @staticmethod
    def _check_group_by_without_index(query: str) -> Optional[Dict[str, str]]:
        """Check GROUP BY columns for index suggestions"""
        group_match = re.search(r"GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)", query.upper(), re.DOTALL)
        if group_match:
            group_cols = QueryOptimizer._extract_columns(group_match.group(1))
            if group_cols:
                cols = [c.lower() for c in group_cols[:3]]
                return {
                    "type": "optimization",
                    "severity": "low",
                    "suggestion": f"Index GROUP BY columns ({', '.join(cols)}) to speed up aggregation",
                    "example": f"CREATE INDEX idx_group ON table({', '.join(cols)})",
                }
        return None

    @staticmethod
    def _check_unbounded_result(query: str) -> Optional[Dict[str, str]]:
        """Check for queries that return potentially large unbounded results"""
        query_upper = query.upper()
        has_agg = any(fn in query_upper for fn in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])
        if "LIMIT" not in query_upper and not has_agg and "GROUP BY" not in query_upper:
            if "WHERE" in query_upper:
                return {
                    "type": "performance",
                    "severity": "low",
                    "suggestion": "Add LIMIT to prevent returning unexpectedly large result sets",
                    "example": "Append 'LIMIT 1000' to cap the number of rows returned",
                }
        return None

    @staticmethod
    def _check_function_on_indexed_column(query: str) -> Optional[Dict[str, str]]:
        """Detect functions applied to columns in WHERE (prevents index usage)"""
        where_match = re.search(r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)", query.upper(), re.DOTALL)
        if where_match:
            clause = where_match.group(1)
            fn_patterns = ["STRFTIME(", "DATE(", "LOWER(", "UPPER(", "SUBSTR(", "CAST("]
            found = [fn.rstrip("(") for fn in fn_patterns if fn in clause]
            if found:
                return {
                    "type": "optimization",
                    "severity": "medium",
                    "suggestion": f"Function ({', '.join(found)}) in WHERE clause prevents index usage",
                    "example": "Use computed/stored columns or restructure the filter to avoid wrapping indexed columns in functions",
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
