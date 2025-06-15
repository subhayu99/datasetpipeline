# 🚀 DatasetPipeline

[![PyPI version](https://badge.fury.io/py/datasetpipeline.svg)](https://badge.fury.io/py/datasetpipeline)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/datasetpipeline)](https://pepy.tech/project/datasetpipeline)

**Transform messy datasets into ML-ready gold.** A powerful, configurable pipeline for dataset processing, quality assessment, and standardization—built by ML practitioner(s), for ML practitioners.

---

## 🎯 Why DatasetPipeline?

**The Problem:** You're drowning in data preprocessing chaos. Multiple formats, inconsistent schemas, duplicate records, quality issues—and you're spending more time wrangling data than training models.

**The Solution:** DatasetPipeline automates the entire journey from raw data to model-ready datasets with reproducible, configurable workflows.

### Born from Real-World Pain 🔥

This project emerged from my experience as a data engineer and MLOps practitioner. I was constantly:

- Ingesting diverse datasets for LLM fine-tuning
- Converting everything to OpenAI-compatible formats
- Writing repetitive preprocessing scripts
- Juggling deduplication, quality checks, and format conversions
- Maintaining brittle pipelines across multiple projects

What started as manageable became overwhelming. DatasetPipeline was built to solve these exact pain points—turning hours of manual work into minutes of configuration.

---

## 🧠 Baked-in Intelligence

One of DatasetPipeline's most powerful features is its **intelligent data understanding**, designed to take the guesswork out of preparing your datasets. Instead of rigid rules or tedious manual mapping, the system comes with **built-in smarts** that allow it to:

- **Automatically Recognize Conversational Roles:** Ever wondered if your "human_utterance" column is the `user` and "bot_reply" is the `assistant`? DatasetPipeline already has a good idea. It's pre-trained to recognize common patterns and automatically map your raw data fields to standard roles like `system`, `user`, `assistant`, `chosen`, and `rejected`. This means less time configuring and more time doing.

- **Intelligently Interpret Complex Structures:** For datasets where conversations are nested in multi-turn formats, DatasetPipeline goes a step further. It can automatically figure out which part of your data represents the `role` (who said it) and which is the `content` (what was said), even when these keys aren't explicitly named or are inconsistent. It's like having a helpful assistant who understands the natural flow of a conversation, regardless of its underlying structure.

- **Adapt to Your Training Needs:** Whether you're fine-tuning a model with single prompt-response pairs (SFT) or training it to prefer one answer over another (DPO), DatasetPipeline adapts its understanding. It tailors the output format to perfectly match the requirements of these different AI training paradigms, ensuring your data is always battle-ready for the task at hand.

- **Anticipate and Validate:** The system isn't just smart about understanding; it's also smart about preventing errors. It includes built-in checks to confirm your data aligns with expected formats, guiding you towards clean, high-quality inputs before you even start training.

In essence, DatasetPipeline aims to be your **intuitive data partner**. It handles the complexities of data interpretation behind the scenes, allowing you to move from raw data to model-ready gold with unprecedented ease and speed.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔌 **Multi-Source Loading** | Hugging Face datasets, local files, cloud storage |
| 🔄 **Format Flexibility** | SFT, DPO, conversational, text—convert between any format |
| 🧹 **Smart Deduplication** | Semantic similarity using embeddings, not just exact matches |
| 📊 **Quality Analysis** | Automated categorization and quality scoring |
| ⚙️ **YAML Configuration** | Reproducible workflows, version-controlled pipelines |
| 🖥️ **CLI Interface** | Simple commands, powerful automation |
| 🚀 **GPU Acceleration** | Optional GPU support for heavy processing |

---

## 🚀 Quick Start

### Installation

```bash
# Recommended: Use as isolated tool
uv tool install datasetpipeline

# Or with pip
pip install datasetpipeline

# For full features (embeddings, GPU support)
pip install "datasetpipeline[all]"
```

### Your First Pipeline

```bash
# Generate a minimal sample configuration with comments
datasetpipeline sample my-first-job.yml --template minimal

# Or generate a full sample with all options and comments
datasetpipeline sample my-first-job.yml --template full

# Run the pipeline
datasetpipeline run my-first-job.yml

# That's it! 🎉
```

---

## ⚙️ Configuration Guidelines

### 🚨 Important Configuration Rule

**When disabling pipeline components, you must keep the section keys present with `null` values. Never completely remove the top-level keys.**

#### ✅ Correct Way to Disable Components

```yaml
load:
  huggingface:
    path: "teknium/OpenHermes-2.5"
    split: "train"

format:
  sft:
    use_openai: true

# Disable deduplication - keep the key with null
deduplicate: null

# Disable analysis - keep the key with null  
analyze: null

save:
  local:
    directory: "output"
    filetype: "jsonl"
```

#### ❌ Wrong Way (Will Cause Errors)

```yaml
load:
  huggingface:
    path: "teknium/OpenHermes-2.5"
    split: "train"

format:
  sft:
    use_openai: true

# DON'T DO THIS - completely removing keys
# deduplicate: <-- missing entirely
# analyze: <-- missing entirely

save:
  local:
    directory: "output"
    filetype: "jsonl"
```

#### 💡 Alternative: Comment Out Values, Keep Keys

```yaml
load:
  huggingface:
    path: "teknium/OpenHermes-2.5"
    split: "train"

format:
  sft:
    use_openai: true

# Temporarily disable deduplication
deduplicate:
  # semantic:
  #   threshold: 0.85
  #   column: "messages"

# Disable analysis for now
analyze:
  # quality:
  #   column_name: "messages"
  #   categories: ["code", "reasoning"]

save:
  local:
    directory: "output"
    filetype: "jsonl"
```

### Why This Matters

DatasetPipeline expects all major pipeline sections (`load`, `format`, `deduplicate`, `analyze`, `save`) to be present in the configuration. This design ensures:

- **Consistent pipeline structure** across all jobs
- **Clear intent** - you explicitly choose to skip steps vs. forgetting them
- **Easy re-enablement** - uncomment values instead of rewriting sections
- **Better error messages** - the pipeline knows what you intended

### 🎛️ Managing Configuration Complexity

**Problem**: The full sample configuration can be overwhelming with all comments and options.

**Solutions**:

1. **Start minimal** - Use `--template minimal` as a starting point for clean, simple configs
2. **Use templates** - Pre-built configurations for common use cases (`--template sft`, `--template dpo`, `--template analysis`)
3. **Progressive enhancement** - Start simple, add complexity as needed
4. **Reference mode** - Use `--template full` when you need to see all available options

---

## 📖 Real-World Example

Transform a Hugging Face dataset into training-ready format:

```yaml
# jobs/sft-training.yml
load:
  huggingface:
    path: "teknium/OpenHermes-2.5"
    split: "train"
    take_rows: 10000

format:
  sft:
    use_openai: true
    column_role_map:
      system: "system"
      human: "user" 
      gpt: "assistant"

deduplicate:
  semantic:
    threshold: 0.85
    column: "messages"

analyze:
  quality:
    column_name: "messages"
    categories: ["code", "reasoning", "creative", "factual"]

save:
  local:
    directory: "training_data"
    filetype: "jsonl"
```

```bash
datasetpipeline run jobs/sft-training.yml
```

**Result:** Clean, deduplicated, standardized training data ready for your LLM fine-tuning pipeline.

---

## 🛠️ Core Commands & Sample Generation

### Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `list` | Preview available jobs | `datasetpipeline list jobs/` |
| `run` | Execute pipeline(s) | `datasetpipeline run jobs/my-job.yml` |
| `sample` | Generate template configs | `datasetpipeline sample new-job.yml --template=minimal` |

### Batch Processing

```bash
# Process all jobs in a directory
datasetpipeline run jobs/

# Preview what will run
datasetpipeline list jobs/
```

---

## 🏗️ Pipeline Components

### 📥 Data Loading

- **Hugging Face**: Direct dataset integration
- **Local Files**: JSON, CSV, Parquet, JSONL
- **Cloud Storage**: S3, GCS (coming soon)

### 🔧 Data Formatting

- **SFT (Supervised Fine-Tuning)**: OpenAI chat format
- **DPO (Direct Preference Optimization)**: Preference pairs
- **Conversational**: Multi-turn dialogue format
- **Text**: Simple text processing
- **Custom Merging**: Combine multiple fields intelligently

### 🧹 Deduplication

- **Semantic**: Embedding-based similarity detection
- **Exact**: Hash-based duplicate removal
- **Fuzzy**: Near-duplicate detection

### 📊 Quality Analysis

- **Automated Categorization**: Code, math, reasoning, creative writing
- **Quality Scoring**: Length, complexity, coherence metrics
- **Custom Categories**: Define your own quality dimensions

### 💾 Data Saving

- **Multiple Formats**: Parquet, JSONL, CSV
- **Flexible Output**: Local files, structured directories
- **Metadata**: Pipeline provenance and statistics

---

## 📁 Project Structure

```
datasetpipeline/
├── 📦 app/
│   ├── 🔬 analyzer/       # Quality analysis & categorization
│   ├── 🧹 dedup/          # Deduplication algorithms
│   ├── 🔄 format/         # Data format transformations
│   ├── 📥 loader/         # Multi-source data loading
│   ├── 💾 saver/          # Output handling
│   └── 🛠️ helpers/        # Utilities & common functions
├── ⚙️ jobs/               # Sample YAML configurations
├── 📊 processed/          # Pipeline outputs
└── 📜 scripts/            # Maintainer utilities
```

---

## 🎨 Advanced Configuration

### Conditional Processing

```yaml
load:
  huggingface:
    path: "my-dataset"
    filters:
      quality_score: ">= 0.8"
      language: "en"

format:
  sft:
    use_openai: true
    min_message_length: 10
    max_conversation_turns: 20

# Skip deduplication for this job
deduplicate: null

analyze:
  quality:
    column_name: "text"
    min_score: 0.7
    categories: ["technical", "creative"]
    save_analysis: true

save:
  local:
    directory: "filtered_data"
    filetype: "parquet"
```

### Quality-Based Filtering

```yaml
load:
  local:
    path: "raw_data.jsonl"

# Skip formatting - data is already in correct format
format: null

deduplicate:
  exact:
    column: "content"

analyze:
  quality:
    column_name: "text"
    min_score: 0.7
    categories: ["technical", "creative"]
    save_analysis: true

save:
  local:
    directory: "cleaned_data"
    filetype: "jsonl"
```

### Custom Deduplication

```yaml
load:
  huggingface:
    path: "my-dataset"

format:
  text:
    column: "content"

deduplicate:
  semantic:
    threshold: 0.9
    model: "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: 32
    use_gpu: true

# Skip analysis for faster processing
analyze: null

save:
  local:
    directory: "deduped_data"
    filetype: "parquet"
```

---

## 🏗️ Extensible Architecture

DatasetPipeline is built with extensibility at its core. Each major component uses **Abstract Base Classes (ABC)**, making it incredibly easy to add new functionality:

```python
# Want a new data loader? Just extend BaseLoader
class MyCustomLoader(BaseLoader):
    def load(self) -> Dataset:
        # Your custom loading logic
        pass

# Need a specialized formatter? Extend BaseFormatter  
class MyFormatter(BaseFormatter):
    def format(self, dataset: Dataset) -> Dataset:
        # Your formatting logic
        pass
```

### 🔌 Pluggable Components

| Component | ABC Base Class | Easy to Add |
|-----------|----------------|-------------|
| 📥 **Loaders** | `BaseLoader` | New data sources (APIs, databases, cloud storage) |
| 🔄 **Formatters** | `BaseFormatter` | Custom data transformations and schemas |
| 🧹 **Deduplicators** | `BaseDeduplicator` | Novel similarity algorithms |
| 📊 **Analyzers** | `BaseAnalyzer` | Domain-specific quality metrics |
| 💾 **Savers** | `BaseSaver` | New output formats and destinations |

### 🚀 Contribution-Friendly

This architecture means:

- **Low barrier to entry**: Add one component without touching others
- **Clean interfaces**: Well-defined contracts between components
- **Easy testing**: Mock and test components in isolation
- **Community growth**: Contributors can focus on their expertise area

**Example**: Want to add PostgreSQL loading? Just implement `BaseLoader` and you're done!

---

## 🏃‍♂️ Performance Tips

- **GPU Acceleration**: Install with `[gpu]` extras for faster embeddings
- **Batch Processing**: Use larger batch sizes for better throughput
- **Memory Management**: Process large datasets in chunks
- **Caching**: Embeddings are cached automatically

```bash
# High-performance setup
pip install "datasetpipeline[gpu]"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** your changes thoroughly
4. **Submit** a pull request

### Development Setup

```bash
git clone https://github.com/subhayu99/datasetpipeline
cd DatasetPipeline
uv pip install -e ".[dev]"
pre-commit install
```

### Areas We Need Help

- 🌐 Cloud storage integrations (S3, GCS, Azure)
- 🔍 Advanced quality metrics
- 📈 Performance optimizations
- 📚 Documentation and examples
- 🧪 Test coverage improvements

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with love by the ML community, for the ML community. Special thanks to all contributors and users who help make dataset preparation less painful.

**Star the repo if DatasetPipeline saves you time!** ⭐

---

Made with ❤️ by [Subhayu Kumar Bala](https://github.com/subhayu99)
