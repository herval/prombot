
# PromQL Cheat Sheet

### 1. **Basic Terminology**

- **Metrics**: Data points (e.g., `http_requests_total`, `node_cpu_seconds_total`).
- **Labels**: Key-value pairs for identifying metrics (e.g., `{job="api", status="200"}`).
- **Time Series**: Metric + labels over time.
- **Instant Vector**: Set of time series at a single timestamp.
- **Range Vector**: Set of time series over a time range.
- **Scalar**: A single numeric value.
- **String**: A single string.

### 2. **Basic Syntax**

- **Instant Vector**: `metric_name{label1="value1", label2!="value2"}`
- **Range Vector**: `metric_name{label="value"}[5m]`
- **Offset**: `metric_name{label="value"} offset 1h`

### 3. **Operators**

- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparison**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical**:
  - `and`: Returns time series that exist in both vectors.
  - `or`: Returns time series that exist in either vector.
  - `unless`: Returns time series from left vector that don’t match the right vector.

### 4. **Aggregation Operators**

- **sum**: Sum of all values (e.g., `sum(metric_name)`).
- **avg**: Average of all values (e.g., `avg(metric_name)`).
- **min**: Minimum value (e.g., `min(metric_name)`).
- **max**: Maximum value (e.g., `max(metric_name)`).
- **count**: Count of elements (e.g., `count(metric_name)`).
- **stddev**: Standard deviation (e.g., `stddev(metric_name)`).
- **stdvar**: Standard variance (e.g., `stdvar(metric_name)`).
- **topk(k, metric_name)**: Top `k` elements by value.
- **bottomk(k, metric_name)**: Bottom `k` elements by value.

> **Grouping Modifiers**: `by` and `without` can be used to specify grouping (e.g., `sum(http_requests_total) by (status)`).

### 5. **Functions**

- **Math**:
  - `abs(v)`: Absolute value.
  - `ceil(v)`: Rounds up.
  - `floor(v)`: Rounds down.
  - `exp(v)`: Exponential.
  - `sqrt(v)`: Square root.
  - `ln(v)`: Natural logarithm.
  - `log10(v)`, `log2(v)`: Base-10 and base-2 logarithms.

- **Time**:
  - `rate(v [range])`: Per-second rate of increase (for counters).
  - `irate(v [range])`: Instantaneous rate (for counters).
  - `increase(v [range])`: Increase in value over the specified range.
  - `delta(v [range])`: Difference over time for gauges.
  - `idelta(v [range])`: Instantaneous delta.
  - `predict_linear(v [range], seconds)`: Predicts value after `seconds`.
  - `holt_winters(v [range], sf, tf)`: Forecasting with smoothing factors.

- **Counting**:
  - `count_over_time(v [range])`: Number of values in a range.
  - `avg_over_time(v [range])`: Average over a range.
  - `min_over_time(v [range])`, `max_over_time(v [range])`: Min/Max over range.
  - `sum_over_time(v [range])`: Sum over range.

- **Other Useful Functions**:
  - `sort(v)`: Sorts in ascending order.
  - `sort_desc(v)`: Sorts in descending order.
  - `label_replace(v, dst_label, replacement, src_label, regex)`: Replaces labels.
  - `label_join(v, dst_label, separator, src_labels...)`: Joins labels.
  - `time()`: Current Unix timestamp.
  - `clamp_min(v, min)`: Clamps to a minimum.
  - `clamp_max(v, max)`: Clamps to a maximum.

### 6. **Recording Rules and Alerts**

- **Recording Rules**: Save complex expressions for reuse.
  ```yaml
  groups:
  - name: example_rules
    rules:
    - record: instance:node_cpu_utilisation:rate5m
      expr: rate(node_cpu_seconds_total{mode!="idle"}[5m])
  ```

- **Alerting Rules**:
  ```yaml
  groups:
  - name: example_alerts
    rules:
    - alert: HighCPUUsage
      expr: rate(node_cpu_seconds_total{mode!="idle"}[5m]) > 0.9
      for: 5m
      labels:
        severity: "critical"
      annotations:
        summary: "High CPU usage detected"
  ```

### 7. **Query Examples**

- **Total HTTP Requests**: `sum(http_requests_total)`
- **Per-Second Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **CPU Usage by Job**: `avg(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (job)`
- **Memory Usage Above Threshold**: `node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes > 1e9`
- **Disk Space Remaining**: `node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}`
- **Top 3 CPU Intensive Jobs**: `topk(3, sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (job))`
- **Alert on High Request Rate**:
  ```promql
  rate(http_requests_total[5m]) > 100
  ```

### 8. **Tips and Best Practices**

- **Label Selection**: Use labels selectively to avoid cardinality explosion.
- **Rate vs. Increase**: Use `rate` for counters, `increase` for cumulative counters over time.
- **Avoid `sum()` over large ranges**: Aggregations over very large datasets may overload the query.
- **Reduce Query Scope**: Use `by` and `without` judiciously to focus results.
- **Downsampling**: Consider `subqueries` or `recording rules` for high-resolution data over time.

### 9. **Advanced Usage**

- **Subqueries**: Nest a query inside a time range.
  ```promql
  sum_over_time(rate(http_requests_total[5m])[1h:5m])
  ```
- **Vector Matching**:
  - **One-to-One Matching**: Default vector matching.
  - **One-to-Many Matching**: Use `ignoring` or `on`.
    ```promql
    http_requests_total{job="api"} / ignoring(status) http_requests_total{job="db"}
    ```
  - **Group-Left / Group-Right Matching**: For asymmetric matches.
    ```promql
    foo * on(instance) group_left(role) bar
    ```
