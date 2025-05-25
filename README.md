# DatasetPipeline

A data processing and analysis pipeline designed to handle various jobs related to data transformation, quality assessment, deduplication, and formatting. The pipeline can be configured and executed using YAML configuration files.

## Features

- **Multi-source data loading**: Load from Hugging Face datasets, local files, and more
- **Flexible data formatting**: Convert between different formats (SFT, DPO, conversational, text)
- **Advanced deduplication**: Semantic deduplication using embeddings
- **Quality analysis**: Automated quality assessment and categorization
- **Configurable pipeline**: YAML-based configuration for reproducible workflows
- **CLI interface**: Easy-to-use command-line interface

## Installation

### From PyPI (Recommended)

```bash
# Use as a uv tool (isolated environment)
uv tool install datasetpipeline

# Or install as a package with pip
pip install datasetpipeline

# Or install as a package with uv
uv pip install datasetpipeline
```

### Optional Dependencies

```bash
# Full embeddings support
pip install "datasetpipeline[full]"

# GPU acceleration
pip install "datasetpipeline[gpu]"

# All features
pip install "datasetpipeline[all]"

# With uv tool
uv tool install "datasetpipeline[full]"
```

## Quick Start

After installation, you can use the CLI tool directly:

```bash
# Check available commands
datasetpipeline --help

# Or use the short alias
dsp --help
```

## Usage

### Listing Jobs

To list all jobs in a pipeline configuration:

```bash
datasetpipeline list jobs/
datasetpipeline list jobs/config.yml
```

### Running the Pipeline

To run a pipeline based on configuration files:

```bash
# Run all jobs in a directory
datasetpipeline run jobs/

# Run a specific job configuration
datasetpipeline run jobs/aeroboros-conv.yml
```

### Generating Sample Configuration

To generate a sample job configuration:

```bash
# Print to stdout
datasetpipeline sample

# Save to file
datasetpipeline sample my-job.yml
datasetpipeline sample my-job.json
```

## Configuration

Job configurations are defined in YAML format. Each configuration specifies the complete pipeline: loading, formatting, deduplication, analysis, and saving.

### Example Configuration

```yaml
# jobs/example-job.yml
load:
  huggingface:
    path: "davanstrien/data-centric-ml-sft"
    split: "train"
    take_rows: 1000

format:
  merger:
    user:
      fields: ["book_id", "author", "text"]
      separator: "\n"
      merged_field: "human"
  sft:
    use_openai: false
    column_role_map:
      persona: "system"
      human: "user"
      summary: "assistant"

deduplicate:
  semantic:
    threshold: 0.8
    column: "messages"

analyze:
  quality:
    column_name: "messages"
    categories: ["code", "math", "science", "literature"]

save:
  local:
    directory: "processed"
    filetype: "parquet"
```

### Configuration Sections

- **`load`**: Configure data sources (Hugging Face, local files)
- **`format`**: Transform data between formats (SFT, DPO, conversational, text)
- **`deduplicate`**: Remove duplicate entries using semantic similarity
- **`analyze`**: Perform quality analysis and categorization
- **`save`**: Save processed data locally or to cloud storage

## Directory Structure

```
app/
├── analyzer/          # Data quality analysis modules
├── dedup/             # Deduplication logic
├── format/            # Data formatting transformations
├── helpers/           # Utility functions and helpers
├── loader/            # Data loading from various sources
├── models/            # Pydantic data models
├── saver/             # Data saving utilities
├── translators/       # Data translation modules
├── cli.py             # CLI entry point
├── constants.py       # Application constants
├── job.py             # Job configuration and execution
├── pipeline.py        # Pipeline orchestration
└── sample_job.py      # Sample configuration

jobs/                  # YAML job configurations (default)
processed/             # Output directory for processed data (default)
scripts/               # Additional utility scripts
```

## Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/subhayu99/datasetpipeline
cd DatasetPipeline
uv pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Running Tests

```bash
pytest
pytest --cov=app  # With coverage
```

### Code Formatting

```bash
black app/
flake8 app/
mypy app/
```

## Optional Dependencies

- **`full`**: Complete embeddings support with transformers
- **`dev`**: Development and testing tools
- **`gpu`**: GPU acceleration for embeddings and deduplication
- **`all`**: All optional dependencies

Install specific groups:
```bash
uv pip install "datasetpipeline[full,gpu]"
```

## Examples

### Basic Text Processing
```bash
# Create a simple job configuration
datasetpipeline sample simple-job.yml

# Edit the configuration as needed
# Then run it
datasetpipeline run simple-job.yml
```

### Batch Processing
```bash
# Process multiple job configurations
datasetpipeline run jobs/

# List all jobs first to preview
datasetpipeline list jobs/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
