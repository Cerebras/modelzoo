# DQSage Data Visualizer

A comprehensive, high-performance data analysis and visualization application designed for large-scale datasets. Features advanced lazy loading, parallel processing, intelligent indexing, and powerful SQL querying capabilities for JSONL and Parquet files.

## 🚀 Key Features

### Core Capabilities
- **Multi-Format Support**: JSONL and Parquet files with format-specific optimizations
- **Lazy Loading**: Memory-efficient batch-based processing with configurable batch sizes (10-1000 records)
- **Parallel Indexing**: Multi-core processing for 4-16x faster performance on JSONL files
- **O(1) Random Access**: Instant batch jumping after indexing (JSONL) or native fast access (Parquet)
- **Advanced SQL Engine**: DuckDB-powered SQL queries with support for nested fields and complex analytics
- **Schema Analysis**: Automatic field detection, nested structure analysis, and data type inference
- **Selective Field Loading**: Load only needed fields to minimize memory usage and improve performance
- **Smart Navigation**: Previous/Next/Jump controls with intelligent boundary protection
- **t-SNE Embedding Visualizer**: 2D / 3D interactive Plotly scatter with reservoir sampling, label distribution analytics (entropy + Jensen–Shannon distance), multi-format metadata + NPZ shard ingestion

### Advanced Features
- **Full Dataset SQL Processing**: Execute SQL queries across entire datasets with export capabilities
- **Batch Range Processing**: Load and query specific ranges of batches for targeted analysis
- **Nested Field Support**: Deep nested field access using dot notation (e.g., `metadata.category.subcategory`)
- **Export Functionality**: Save query results to JSONL or CSV with optional chunking

## ⚡ Performance Recommendations

### File Format Priority
1. **Parquet (Highly Recommended)**: 
   - 5-10x faster than JSONL
   - No indexing required
   - Native columnar storage for analytics
   - Optimal for cross-file queries

2. **JSONL**: 
   - Requires indexing for optimal performance
   - Great for streaming data

### Data Conversion Guidelines
Convert data to Parquet for maximum performance:
- `.json.gz` → Parquet (10x speed improvement)
- `.csv.gz` → Parquet (5x speed improvement)
- `.zst` files → Parquet (significant compression gains)
- Large JSONL files → Parquet (faster queries, lower memory usage)

## 📋 Requirements & Installation

### System Requirements
- **Python**: 3.11.8+
- **Memory**: 8GB+ RAM (16GB+ recommended for large datasets)
- **CPU**: Multi-core CPU recommended for parallel indexing 
- **Storage**: SSD recommended for best I/O performance with large datasets

### Quick Start Installation
```bash
# Clone or download the DQSage directory
cd dqsage

# Create a virtual environment with Python 3.11.8
python3.11 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run main.py
```

### Dependencies
```bash
# Core Data Processing and Visualization
streamlit==1.47.0      # Modern web app framework
pandas==2.3.0          # Data manipulation and analysis
numpy==2.3.0           # Numerical computing

# High-Performance SQL Engine
duckdb==1.3.2         # Ultra-fast analytics database
sqlparse==0.5.0        # SQL query parsing and validation

# Advanced File Format Support
pyarrow==21.0.0        # Parquet file processing

# Machine Learning (t-SNE)
scikit-learn==1.5.0    # Provides TSNE implementation
scipy==1.11.0          # SciPy stack (neighbors / distances)

# Visualization Libraries
altair==5.5.0          # Declarative statistical visualization (core app)
plotly==5.22.0         # Interactive 2D / 3D embedding scatter & bar charts

# High-performance I/O & Compression
zstandard==0.22.0      # .jsonl.zst decompression for metadata
orjson==3.11.2         # Fast JSON parsing for large metadata streams
```

**Built-in Modules**: The application uses Python's standard library extensively (json, glob, os, datetime, re, collections, time, traceback, multiprocessing, concurrent.futures, functools, atexit, uuid) - no additional installation required.

## 🚨 Important Requirements & Data Structure Guidelines

### Critical Data Structure Requirements
- **IDENTICAL STRUCTURE MANDATORY**: ALL files within a directory MUST have the exact same field structure
- **Field Consistency**: Field names, types, and nested structures must match across all files
- **Nested Field Uniformity**: All nested objects must have consistent key structures
- **No Mixed Formats**: Do not mix JSONL and Parquet files in the same directory

### Supported File Organization
```
your_data_project/
├── jsonl_dataset/              # ✅ All JSONL files with same structure
│   ├── chunk_001.jsonl
│   ├── chunk_002.jsonl
│   ├── chunk_003.jsonl
│   └── ...
├── parquet_dataset/            # ✅ All Parquet files with same structure  
│   ├── batch_001.parquet
│   ├── batch_002.parquet
│   ├── batch_003.parquet
│   └── ...
├── code_data/                  # ✅ Programming code dataset
│   ├── repositories_001.jsonl
│   ├── repositories_002.jsonl
│   └── ...
├── math_data/                  # ✅ Mathematical content dataset
│   ├── problems_001.parquet
│   ├── problems_002.parquet
│   └── ...
└── mixed_format/               # ❌ NOT SUPPORTED - will cause errors
    ├── data.jsonl
    └── data.parquet
```

## 📖 Complete Feature Guide & Usage Instructions

### Overview of Application Interface
The DQSage Data Visualizer provides a comprehensive web-based interface with the following main sections:
1. **Sidebar Configuration**: Data source setup and loading configuration
2. **Schema Analysis**: Field discovery and structure validation
3. **Lazy Loading Controls**: Batch management and indexing
4. **Data Explorer Tabs**: Raw data viewing and SQL querying
5. **Export & Analysis Tools**: Full dataset processing and export capabilities
6. **t-SNE Embedding Visualizer**: Dimensionality reduction & cluster quality diagnostics

---

## 🔧 Step-by-Step Usage Guide

### Phase 1: Initial Setup and Configuration

#### 1.1 Launch the Application
```bash
streamlit run main.py

# The application will open in your default web browser at http://localhost:8501
```

#### 1.2 Configure Data Source (Sidebar)
1. **Set Data Directory Path**:
   - Enter the full path to your data directory (e.g., `/path/to/your/data` or `./my_dataset`)
   - The directory should contain files of the same format and structure
   - Examples: `data/`, `/home/user/datasets/code_data/`, `C:\Data\analysis\`

2. **Select File Format**:
   - Choose between available formats: `JSONL` or `Parquet`
   - The application shows format availability status
   - Parquet is recommended for best performance

3. **Configure File Pattern**:
   - Specify the pattern to match your files
   - Common patterns:
     - `*.jsonl` (all JSONL files)
     - `chunk_*.jsonl` (specific prefix)
     - `*.parquet` (all Parquet files)
     - `data_*.parquet` (with prefix)
     - `file_name.jsonl` (single JSONL files)
     - `file_name.parquet` (single Parquet files)

---

### Phase 2: Schema Discovery and Analysis

#### 2.1 Analyze Data Schema
1. **Click "🔍 Analyze Data Schema"** in the sidebar
   - The system examines the first record from the first file
   - Automatically detects all available fields and nested structures
   - Validates file format compatibility

2. **Review Schema Analysis Results**:
   - **Files Found**: Number of files matching your pattern
   - **File Format**: Confirmed format type
   - **Total Available Fields**: Count of all discoverable fields
   - **Top-level Fields**: Direct fields in the root object
   - **Nested Dictionary Fields**: Fields within nested objects

#### 2.2 Understand Nested Field Support
The application supports unlimited nesting depth:
- **Simple fields**: `id`, `text`, `timestamp`
- **One-level nesting**: `metadata.language`, `stats.word_count`
- **Deep nesting**: `metadata.details.category.subcategory.type`
---

### Phase 3: Field Selection and Loader Configuration

#### 3.1 Select Fields for Loading
1. **Field Selection Benefits**:
   - **Memory Efficiency**: Load only needed fields to reduce RAM usage
   - **Performance**: Faster processing with fewer fields
   - **Focus**: Concentrate analysis on relevant data

2. **Field Selection Process**:
   - Use the multiselect dropdown in the sidebar
   - Choose from all available fields (including nested ones)
   - Default selection includes `id` and `text` if available
   - Select 5-20 fields for optimal performance

#### 3.2 Configure Batch Loading Settings
1. **Batch Size Selection**:
   - **Small (10-50)**: For memory-constrained environments or very wide records
   - **Medium (100-200)**: Balanced performance for most use cases (recommended)
   - **Large (500-1000)**: For high-memory systems and narrow records

2. **Memory Considerations**:
   - Larger batches = faster processing but more memory usage
   - Smaller batches = slower processing but lower memory footprint
   - Monitor memory usage displayed in the interface

#### 3.3 Initialize Lazy Data Loader
1. **Click "🚀 Initialize Lazy Data Loader"**
   - Creates the lazy loading system with your configuration
   - Automatically loads the first batch for immediate preview
   - Clears any previous loader state and caches
---

### Phase 4: High-Performance Indexing (JSONL Files Only)

#### 4.1 Understanding Indexing Benefits
- **For JSONL Files**: Indexing is highly recommended for datasets > 10,000 records
- **For Parquet Files**: No indexing needed - DuckDB provides native fast access
- **Performance Impact**: Up to 100x faster random access after indexing

#### 4.2 Build Performance Index (JSONL)
1. **Configure Indexing Options**:
   - **Parallel Processing**: Enable for 4-16x faster indexing (recommended)
   - **Worker Count**: Set to number of CPU cores (auto-detected)
   - **Progress Tracking**: Monitor real-time indexing progress

2. **Execute Indexing**:
   - Click "🔧 Build Index for Fast Access"
   - Watch real-time progress with file-by-file updates
   - See performance metrics and throughput statistics

3. **Index Benefits**:
   - **O(1) Batch Lookup**: Instant access to any batch number
   - **Precise Navigation**: Jump to exact batch locations
   - **Memory Efficient**: Index overhead is minimal (~200 bytes per batch)

#### 4.3 Index Management
- **View Index Statistics**: Batch count, memory usage, file coverage
- **Clear Index**: Remove index to free memory or rebuild with different settings
- **Operating Modes**:
  - **Indexed Mode**: Fast random access, precise navigation
  - **Sequential Mode**: Slower but works without indexing

---

### Phase 5: Batch Navigation Controls
1. **Basic Navigation**:
   - **Previous Batch (⏮️)**: Load the previous batch in sequence
   - **Next Batch (⏭️)**: Load the next batch in sequence
   - **Current Batch Display**: Shows current position and total batches

2. **Advanced Navigation**:
   - **Jump to Batch**: Enter specific batch number for direct access
   - **Reset to First**: Return to batch 0 and clear cache


### Phase 6: Raw Data Exploration

#### 6.1 Raw Data Viewer Features
   1. **Full Data View**: Complete batch data in table format
   2. **Column Selection**: Choose which fields to display
   3. **Scrollable Interface**: Handle wide datasets efficiently
   4. **Data Type Detection**: Automatic field type recognition
   5. **Search Capability**: The raw data viewer supports searching across fields for quick inspection and filtering


#### 6.2 Individual Record Inspection
   1. **JSON Format Display**: Pretty-printed record structure
   2. **Nested Field Expansion**: Drill down into complex nested data
   3. **Field Navigation**: Easy exploration of deep data structures
   4. **Record Selection**: Choose specific records for detailed examination


---

### Phase 7: Advanced SQL Querying

#### 7.1 SQL Query Scope Options

##### 7.1.1 Current Batch Queries
- **Use Case**: Quick analysis of currently loaded data
- **Performance**: Instant execution (data already in memory)
- **Limitations**: Limited to current batch size (10-1000 records)

**Example Current Batch Queries**:
```sql
-- Basic data exploration
SELECT * FROM data LIMIT 10

-- Field analysis
SELECT COUNT(*) as total_records, 
       COUNT(DISTINCT id) as unique_ids 
FROM data

-- Text analysis
SELECT AVG(LENGTH(text)) as avg_text_length,
       MAX(LENGTH(text)) as max_text_length
FROM data
```

##### 7.1.2 Batch Range Queries
- **Use Case**: Analysis across multiple consecutive batches
- **Performance**: Fast loading with optimized range access
- **Scope**: Configurable start and end batch numbers

**Steps for Batch Range Queries**:
1. **Configure Range**: Set start batch and end batch numbers
2. **Load Range**: Click "📥 Load Batch Range for SQL"
3. **Execute Queries**: Run SQL on the combined dataset

**Example Batch Range Queries**:
```sql
-- Cross-batch analysis
SELECT source_file, COUNT(*) as records_per_file
FROM data 
GROUP BY source_file
ORDER BY records_per_file DESC

-- Data quality assessment
SELECT 
    COUNT(*) as total_records,
    COUNT(CASE WHEN text IS NOT NULL THEN 1 END) as valid_text,
    COUNT(CASE WHEN LENGTH(text) > 100 THEN 1 END) as substantial_text
FROM data
```

##### 7.1.3 Full Dataset Queries with Export
1. **Direct File Processing**: Query raw files without loading into memory
2. **Export Options**: Save results as JSONL or CSV
3. **Chunking Support**: Split large results into multiple files

#### 7.2 Full Dataset SQL Export Workflow

##### 7.2.1 Query Configuration
1. **Write SQL Query**: Use DuckDB-compatible SQL syntax
2. **Configure Output**:
   - **Output Directory**: Where to save results
   - **File Format**: JSONL or CSV
   - **Filename Options**: Add timestamps, chunking

3. **Advanced Options**:
   - **Chunking**: Split results for memory efficiency
   - **Optimization**: Use single-query mode for best performance

##### 7.2.2 Export Examples

**Basic Data Filtering and Export**:
```sql
-- Export high-quality records
SELECT id, text, "metadata.language", "metadata.score"
FROM data 
WHERE "metadata.score" > 0.7 
  AND LENGTH(text) > 200
```

**Advanced Data Transformation**:
```sql
-- Create enriched dataset
SELECT 
    id,
    text,
    "metadata.language" as language,
    "metadata.score" as quality_score,
    LENGTH(text) as text_length,
    CASE 
        WHEN "metadata.category" = 'code' THEN 'Programming'
        WHEN "metadata.category" = 'math' THEN 'Mathematics'
        ELSE 'General'
    END as content_type,
    CURRENT_TIMESTAMP as export_timestamp
FROM data 
WHERE "metadata.verified" = true
ORDER BY quality_score DESC
```

---

### Phase 8: Performance Optimization & Best Practices

#### 8.1 Performance Optimization Guidelines

##### 8.1.1 For JSONL Files
1. **Always Build Index**: Essential for datasets > 10,000 records
2. **Use Parallel Indexing**: Leverage all CPU cores for 4-16x speedup
3. **Optimize Field Selection**: Load only necessary fields
4. **Consider Parquet Conversion**: For repeated analysis on large datasets

##### 8.1.2 For Parquet Files
1. **No Indexing Required**: DuckDB provides native optimization
2. **Leverage Column Storage**: Parquet's columnar format is ideal for analytics
3. **Direct Directory Queries**: System automatically optimizes cross-file access
4. **Batch Operations**: Large batches work efficiently with Parquet

##### 8.1.3 General Optimization
1. **Memory Management**:
   - Monitor cache usage and clear when needed
   - Adjust batch sizes based on available RAM
   - Use field selection to reduce memory footprint

2. **Query Optimization**:
   - Use appropriate data scopes (Current Batch vs Full Dataset)
   - Leverage indexing for random access patterns
   - Prefer Full Dataset mode for large analytical queries

#### 8.2 Troubleshooting Common Issues

##### 8.2.1 Performance Issues
**Problem**: Slow data loading
- **Solution**: Build index for JSONL files, convert to Parquet, or reduce batch size

**Problem**: High memory usage
- **Solution**: Clear cache, reduce batch size, select fewer fields

**Problem**: Slow SQL queries
- **Solution**: Use appropriate query scope, add WHERE clauses, consider indexing

##### 8.2.2 Data Structure Issues
**Problem**: "Mixed file structures detected"
- **Solution**: Ensure all files have identical schema, separate different structures

**Problem**: Nested field access errors
- **Solution**: Use double quotes around field names, verify field paths in schema analysis

**Problem**: File format errors
- **Solution**: Verify file format selection matches actual files, check file patterns

##### 8.2.3 System Issues
**Problem**: Application crashes or freezes
- **Solution**: Restart application, clear session state, check system memory

**Problem**: File not found errors
- **Solution**: Verify directory paths, check file permissions, ensure pattern matching

---

## 🧪 Use Case Examples

### Use Case 1: Code Repository Analysis
**Scenario**: Analyze a large corpus of programming code files

```bash
# Data structure example
{
  "id": "repo_12345_file_678",
  "text": "def process_data(input_file):\n    # Function implementation...",
  "metadata": {
    "language": "python",
    "repository": "tensorflow/tensorflow",
    "file_path": "tensorflow/python/ops/array_ops.py",
    "lines_of_code": 150,
    "complexity_score": 0.72
  }
}
```

**Analysis Workflow**:
1. **Load code dataset** (select fields: id, text, metadata.language, metadata.complexity_score)
2. **Build index** for fast navigation across millions of code files
3. **Language analysis**:
   ```sql
   SELECT "metadata.language", 
          COUNT(*) as file_count,
          AVG("metadata.complexity_score") as avg_complexity
   FROM data 
   GROUP BY "metadata.language"
   ORDER BY file_count DESC;
   ```
4. **Export high-quality code samples**:
   ```sql
   SELECT id, text, "metadata.language", "metadata.complexity_score"
   FROM data 
   WHERE "metadata.complexity_score" > 0.8 
     AND LENGTH(text) > 500;
   ```

### Use Case 2: Mathematical Content Processing
**Scenario**: Process educational mathematics problems and solutions

```bash
# Data structure example
{
  "id": "math_problem_9876",
  "text": "Solve the integral: ∫(x² + 2x + 1)dx from 0 to 5",
  "metadata": {
    "subject": "calculus",
    "difficulty": "intermediate",
    "solution_steps": 4,
    "verified": true,
    "quality_score": 0.91
  }
}
```

**Analysis Workflow**:
1. **Schema analysis** to discover all mathematical metadata fields
2. **Field selection** focusing on content and quality metrics
3. **Difficulty-based analysis**:
   ```sql
   SELECT "metadata.difficulty",
          COUNT(*) as problem_count,
          AVG("metadata.quality_score") as avg_quality,
          AVG("metadata.solution_steps") as avg_steps
   FROM data 
   GROUP BY "metadata.difficulty"
   ORDER BY avg_quality DESC;
   ```

### Use Case 3: Large-Scale Text Analysis
**Scenario**: Analyze millions of documents for content research

**Performance Strategy**:
1. **Convert to Parquet** for optimal performance across terabytes of data
2. **Full dataset SQL queries** for comprehensive analysis:
   ```sql
   SELECT 
       CASE 
           WHEN LENGTH(text) < 1000 THEN 'Short'
           WHEN LENGTH(text) < 5000 THEN 'Medium'
           ELSE 'Long'
       END as document_type,
       "metadata.source",
       COUNT(*) as count,
       AVG("metadata.quality_score") as avg_quality
   FROM data 
   WHERE "metadata.quality_score" > 0.5
   GROUP BY document_type, "metadata.source"
   ORDER BY count DESC
   ```
4. **Export results** in chunks for downstream processing

---

## 🌀 Phase 9: t-SNE Embedding Visualization (Embeddings Explorer)

The t-SNE module is a second application accessible via the unified launcher. It lets you uniformly sample large labelled corpora, align them with pre‑computed embedding shards, project to 2D / 3D using scikit-learn's TSNE, and analyze cluster label distributions.

### 9.1 Launching the Multi‑App Suite
```bash
streamlit run main.py
```
In the left sidebar (Application Module) choose: **t-SNE Visualizer**.

### 9.2 Required Directory Structure
```
<root>/
   metadata/                # (labelled_dir) JSONL metadata files
      chunk_00001.jsonl[.gz|.zst]
      ...
   embeddings/              # (embedding_dir) NPZ shards
      shard_0.npz
      shard_1.npz
      ...
```
Each metadata line must contain (or allow dot‑path extraction of):
- id (string / numeric convertible)
- optional text (for hover)  
- optional cluster identifier (e.g. cluster_id, label, topic)

Each embedding shard `*.npz` must include:
- `doc_ids` (1-D array of ids as stored in metadata, order arbitrary)
- `embeddings` (2-D float array shape `[N, D]`)

### 9.3 Supported Metadata Formats
Automatic detection / streaming across: `.jsonl`, `.jsonl.gz`, `.jsonl.zst` (zstandard). Files are scanned in sorted order; sampling is streaming & memory efficient.

### 9.4 Reservoir Sampling (Uniform)
- True single-pass reservoir sampling → unbiased selection across arbitrarily large corpora.
- Parameter: **Sample Size** (100 – 200,000). Typical sweet spots:
   - 2,000–5,000: fast exploratory plots
   - 10,000–20,000: richer structure (heavier runtime / memory)
- Complexity: O(total_lines) streaming read with O(sample_size) memory.

### 9.5 Embedding Alignment
After sampling metadata ids, the tool scans NPZ shards until all sampled ids are collected (early-exit when complete). Missing ids are silently dropped (sample shrinks); this avoids bias as long as embeddings were precomputed for most examples.

### 9.6 t-SNE Parameters (Sidebar)
- Dimension: 2D or 3D (3D uses Plotly Scatter3d; heavier to render)
- Perplexity: 5–100 (rule-of-thumb: `< sample_size / 3`)
- Iterations (max_iter / n_iter): 250–5000 (default 750)
- Learning Rate: auto or manual (10–1000)
- Seed: deterministic reproducibility (fixed to 42 for sampling + TSNE random_state)
- Extra Hover Field: optional dot path appended to hover tooltip

### 9.7 Performance Tips
- t-SNE scales roughly O(N²) (pairwise affinities). Keep `sample_size` moderate.
- Increase iterations only after you verify coarse structure.
- Start with 2D; upgrade to 3D once clusters look meaningful.
- If clusters are very large & dense, consider reducing sample size.

### 9.8 Interactive Plot Features
- Per-cluster legend toggle (Select All / Deselect All buttons)
- Hover tooltips with multi-line wrapped text + cluster + extra field
- 3D rotation & zoom (when 3D selected)
- Stable UI state (figure retained until re-run)

### 9.9 Label Distribution Analytics
Separate **Distribution** tab computes full dataset cluster frequencies (not just the sample):
- Full scan across all metadata files (cached; invalidated if files change)
- Outputs top-K clusters + aggregated OTHERS row
- Metrics: total records, unique clusters, top cluster %, entropy (bits)
- Optional comparison with current sample → Jensen–Shannon Distance (bits)
- Log-scale toggle for skewed distributions

### 9.10 Handling Missing / Irregular Labels
- Blank / null / 'none' / 'null' → mapped to `UNKNOWN`
- All labels normalized to strings for consistent grouping.

### 9.11 Field Path Overrides
Auto-detected top-level + first-level nested keys are offered. For deeper nesting use manual dot-path overrides (e.g. `metadata.details.category`). Extraction gracefully returns `None` if path missing.

### 9.12 Typical Workflow
1. Choose metadata + embedding directories.
2. Adjust sample size & perplexity (start small: 2000 / 30).
3. (Optional) Specify cluster_id & extra hover field.
4. Run t-SNE → inspect scatter; iterate parameters.
5. Open Distribution tab → compute full label distribution & compare sample bias (JSD).

### 9.13 Troubleshooting
| Issue | Cause | Mitigation |
|-------|-------|------------|
| No embeddings collected | id mismatch between metadata & NPZ `doc_ids` | Ensure consistent string casting; regenerate shards |
| Perplexity error | Perplexity >= sample_size | Reduce perplexity or increase sample |
| Slow run | Sample too large / high iterations | Lower sample / iterations; use 2D first |
| Many UNKNOWN labels | Missing cluster field | Set correct cluster field or override path |

---
