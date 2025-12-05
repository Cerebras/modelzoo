# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SQL query processor using DuckDB for high-performance data analysis.
Provides secure SQL query execution with caching and validation.
"""

import re
import threading
import time

import duckdb
import sqlparse


class SQLQueryProcessor:
    """High-performance SQL query processor using DuckDB for data analysis"""

    def __init__(self):
        self.connection = None
        self.query_cache = {}
        self.max_cache_size = 50
        self._connection_lock = threading.Lock()
        # Removed max_result_rows limit - no restriction on query results

    def get_connection(self):
        """Get or create DuckDB connection (thread-safe)"""
        if self.connection is None:
            with self._connection_lock:
                self.connection = duckdb.connect(':memory:')
        return self.connection

    def validate_sql_query(self, query):
        """Validate SQL query syntax and check for dangerous operations using proper SQL parsing"""
        try:
            # Parse the SQL query into AST
            parsed = sqlparse.parse(query)
            if not parsed:
                return False, "Empty or invalid SQL query"

            # Check if we have any meaningful statements
            has_meaningful_statements = False

            # Analyze each statement in the query
            for statement in parsed:
                # Skip empty or comment-only statements
                meaningful_tokens = [
                    token
                    for token in statement.flatten()
                    if not token.is_whitespace
                    and token.ttype
                    not in (
                        sqlparse.tokens.Comment.Single,
                        sqlparse.tokens.Comment.Multiline,
                    )
                ]

                if not meaningful_tokens:
                    continue  # Skip comment-only or empty statements

                has_meaningful_statements = True

                # Get the first meaningful token
                first_token = meaningful_tokens[0]

                # Check if it's a SELECT statement or WITH (CTE)
                if (
                    first_token.ttype is sqlparse.tokens.Keyword.DML
                    and first_token.value.upper() == 'SELECT'
                ):
                    # This is a SELECT - need to check for nested dangerous operations
                    if self._contains_dangerous_operations(statement):
                        return (
                            False,
                            "Query contains nested dangerous operations",
                        )
                    continue  # This SELECT is allowed
                elif (
                    first_token.ttype
                    in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.CTE)
                    and first_token.value.upper() == 'WITH'
                ):
                    # CTE (Common Table Expression) - allowed if it doesn't contain dangerous operations
                    if self._contains_dangerous_operations(statement):
                        return False, "CTE contains dangerous operations"
                    continue  # CTE is allowed
                elif first_token.ttype is sqlparse.tokens.Keyword:
                    # Any other SQL keyword at statement start is potentially dangerous
                    keyword = first_token.value.upper()
                    dangerous_keywords = [
                        'DROP',
                        'DELETE',
                        'INSERT',
                        'UPDATE',
                        'ALTER',
                        'CREATE',
                        'TRUNCATE',
                        'MERGE',
                        'CALL',
                        'EXEC',
                        'EXECUTE',
                        'GRANT',
                        'REVOKE',
                        'COMMIT',
                        'ROLLBACK',
                        'SET',
                        'USE',
                        'EXPLAIN',
                        'ANALYZE',
                    ]
                    if keyword in dangerous_keywords:
                        return (
                            False,
                            f"Dangerous operation '{keyword}' not allowed. Only SELECT queries are supported.",
                        )
                    else:
                        return (
                            False,
                            f"Unsupported statement type '{keyword}'. Only SELECT queries are supported.",
                        )
                else:
                    # Statement doesn't start with a recognized keyword
                    return (
                        False,
                        "Invalid SQL statement. Only SELECT queries are supported.",
                    )

            # If we parsed successfully but found no meaningful statements
            if not has_meaningful_statements:
                return False, "No meaningful SQL statements found"

            return True, "Query is valid"

        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"

    def _contains_dangerous_operations(self, statement):
        """Check if a statement contains dangerous operations in subqueries or nested contexts"""
        dangerous_keywords = [
            'DROP',
            'DELETE',
            'INSERT',
            'UPDATE',
            'ALTER',
            'CREATE',
            'TRUNCATE',
            'MERGE',
            'CALL',
            'EXEC',
            'EXECUTE',
            'GRANT',
            'REVOKE',
            'COMMIT',
            'ROLLBACK',
        ]

        # Flatten and check all tokens in the statement
        for token in statement.flatten():
            if (
                token.ttype
                in (
                    sqlparse.tokens.Keyword.DML,
                    sqlparse.tokens.Keyword.DDL,
                    sqlparse.tokens.Keyword,
                )
                and token.value.upper() in dangerous_keywords
            ):
                return True
        return False

    def register_dataframe(self, df, table_name='data'):
        """Register pandas DataFrame as virtual table in DuckDB"""
        try:
            conn = self.get_connection()

            # Drop existing table if it exists
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass

            # Register the DataFrame
            conn.register(table_name, df)
            return (
                True,
                f"DataFrame registered as table '{table_name}' with {len(df):,} rows",
            )

        except Exception as e:
            return False, f"Error registering DataFrame: {str(e)}"

    def execute_query(self, query, table_name='data'):
        """Execute SQL query and return results"""
        try:
            # Validate query first
            is_valid, validation_msg = self.validate_sql_query(query)
            if not is_valid:
                return None, validation_msg

            conn = self.get_connection()
            start_time = time.time()

            # Check cache first
            cache_key = f"{query}_{table_name}"
            if cache_key in self.query_cache:
                cached_result, cached_time, cached_rows = self.query_cache[
                    cache_key
                ]
                return (
                    cached_result,
                    f"✅ Query executed (cached) in {cached_time:.3f}s - {cached_rows:,} rows",
                )

            # Execute query directly without row limit
            result_df = conn.execute(query).fetchdf()
            execution_time = time.time() - start_time

            # Cache the result if it's not too large
            if len(result_df) < 1000:  # Only cache smaller results
                if len(self.query_cache) >= self.max_cache_size:
                    # Remove oldest cached query
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]

                self.query_cache[cache_key] = (
                    result_df.copy(),
                    execution_time,
                    len(result_df),
                )

            success_msg = f"✅ Query executed in {execution_time:.3f}s - {len(result_df):,} rows"

            return result_df, success_msg

        except Exception as e:
            error_msg = f"❌ Query execution error: {str(e)}"
            return None, error_msg

    def get_table_info(self, table_name='data'):
        """Get information about registered table"""
        try:
            conn = self.get_connection()

            # Get column information
            columns_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()

            # Get row count
            count_result = conn.execute(
                f"SELECT COUNT(*) as row_count FROM {table_name}"
            ).fetchdf()
            row_count = count_result['row_count'].iloc[0]

            return {
                'columns': columns_df.to_dict('records'),
                'row_count': row_count,
                'table_exists': True,
            }

        except Exception as e:
            return {
                'columns': [],
                'row_count': 0,
                'table_exists': False,
                'error': str(e),
            }

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()

    def close_connection(self):
        """Close DuckDB connection (thread-safe)"""
        with self._connection_lock:
            if self.connection:
                self.connection.close()
                self.connection = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed"""
        self.close_connection()

    def __del__(self):
        """Destructor - ensures connection is closed when object is garbage collected"""
        try:
            self.close_connection()
        except:
            # Ignore errors during cleanup
            pass

    @staticmethod
    def standardize_sql_syntax(query, type_map=None):
        """Transform SQL queries for Full Dataset mode to handle raw JSON access properly.
        - Converts quoted nested fields like "a.b.c" to a->>'b'->>'c'
        - Optionally wraps nested JSON scalars with TRY_CAST(... AS <TYPE>) when type_map is provided.

        Params:
          query: str - user SQL
          type_map: Optional[dict[str, str]] - mapping of column name to DuckDB type (e.g., 'BIGINT')
        """
        if type_map is None:
            type_map = {}

        def pick_cast_target(sql_type: str | None):
            if not sql_type:
                return None
            t = str(sql_type).upper()
            if any(
                x in t
                for x in [
                    "STRUCT",
                    "LIST",
                    "MAP",
                    "JSON",
                    "VARCHAR",
                    "TEXT",
                    "BLOB",
                ]
            ):
                return None
            if any(
                x == t
                for x in [
                    "BIGINT",
                    "HUGEINT",
                    "INTEGER",
                    "INT",
                    "SMALLINT",
                    "TINYINT",
                    "UBIGINT",
                    "UINTEGER",
                    "USMALLINT",
                    "UTINYINT",
                ]
            ):
                return "BIGINT"
            if any(
                x in t
                for x in ["DOUBLE", "REAL", "FLOAT", "DECIMAL", "NUMERIC"]
            ):
                return "DOUBLE"
            if "BOOLEAN" in t:
                return "BOOLEAN"
            if "TIMESTAMP" in t:
                return "TIMESTAMP"
            if t == "DATE":
                return "DATE"
            if t == "TIME":
                return "TIME"
            return None

        def transform_nested_field(match):
            field_path = match.group(1)
            parts = field_path.split('.')
            if len(parts) == 1:
                return f'"{field_path}"'
            base_field = parts[0]
            nested_parts = parts[1:]
            json_expr = base_field
            for part in nested_parts:
                json_expr += f"->>{repr(part)}"
            cast_target = pick_cast_target(type_map.get(field_path))
            if cast_target:
                return f"TRY_CAST({json_expr} AS {cast_target})"
            return json_expr

        pattern = r'"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)"'
        transformed_query = re.sub(pattern, transform_nested_field, query)
        return transformed_query
