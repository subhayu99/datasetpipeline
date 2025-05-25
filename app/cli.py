"""
CLI entry point for DatasetPipeline.
"""

import rich
from pathlib import Path
from typing import Optional

import typer
from app import Pipeline, JobConfig
from app.sample_job import get_config_by_type, TemplateType


# Main CLI app
app = typer.Typer(
    name="datasetpipeline",
    help="A data processing and analysis pipeline for dataset transformation, quality assessment, deduplication, and formatting.",
    no_args_is_help=True
)

def load_pipeline_from_path(path: str):
    path: Path = Path(path)
    if path.is_dir():
        pipeline = Pipeline.from_dir(path)
    elif path.is_file():
        try:
            pipeline = Pipeline.from_file(path)
        except Exception:
            pipeline = Pipeline(jobs=[])
        try:
            configs = [JobConfig.from_file(path)]
        except Exception:
            configs = []
        
        pipeline.jobs = pipeline.jobs + configs
    else:
        raise ValueError("Invalid path. Must be a directory or a file.")
    return pipeline


@app.command(name="list", help="List all jobs in the pipeline.")
def list_jobs(
    path: str = typer.Argument(..., help="Path to load config(s) from. Can be a directory or a file."),
):
    pipeline = load_pipeline_from_path(path)
    rich.print(f"Total jobs: {pipeline.total_jobs}")
    for job in pipeline.get_jobs():
        rich.print(job.config, end="\n\n")
    

@app.command(name="run", help="Run a pipeline from config(s).")
def run_pipeline(
    path: str = typer.Argument(..., help="Path to load config(s) from. Can be a directory or a file."),
):
    pipeline = load_pipeline_from_path(path)
    pipeline.run()

@app.command(help="Generate sample job configurations with different levels of detail.")
def sample(
    file: Optional[str] = typer.Argument(None, help="Path to save the config to. Can have .json or .yaml extension. Prints to stdout if not specified."),
    template: Optional[TemplateType] = typer.Option(
        TemplateType.FULL, 
        "-t",
        "--template", 
        help="Use a specific template: 'minimal' for a minimal clean config, 'sft' for supervised fine-tuning, 'dpo' for direct preference optimization, 'analysis' for quality analysis focused or 'full' for a comprehensive reference."
    ),
):
    """
    Generate sample configurations for different use cases.
    
    Examples:
        # Quick start - minimal clean config
        datasetpipeline sample quickstart.yml
        
        # Full reference with all options
        datasetpipeline sample reference.yml --template=full
        
        # SFT training template
        datasetpipeline sample sft-job.yml --template=sft
        
        # DPO training template  
        datasetpipeline sample dpo-job.yml --template=dpo
    """
    
    # Get the appropriate configuration
    config_to_use = get_config_by_type(template)
    
    # Output configuration
    if file is None:
        # Print to stdout
        rich.print("[green]Generated configuration:[/green]", end="\n\n")
        
        if template:
            rich.print(f"[yellow]Template: {template.value}[/yellow]", end="\n\n")
            
        # You'll need to modify your config classes to support a comments parameter
        # For now, just output the config
        rich.print(config_to_use, end="\n\n")
        
        # Add helpful next steps
        rich.print("[dim]💡 Tip: Save this config to a file and modify as needed:[/dim]")
        rich.print("[dim]   datasetpipeline sample my-job.yml[/dim]")
        rich.print("[dim]   datasetpipeline run my-job.yml[/dim]")
    else:
        # Save to file
        file_path = Path(file)
        if file_path.exists():
            if not typer.confirm(f"File {file} already exists. Overwrite?", abort=True):
                return
        
        # Save the configuration
        config_to_use.to_file(file)
        
        # Provide feedback
        rich.print(f"[green]✓[/green] Generated {template.value} configuration: [blue]{file}[/blue]")
        
        # Show next steps
        rich.print("[dim]Next steps:[/dim]")
        rich.print(f"[dim]  1. Edit {file} to customize for your needs[/dim]")
        rich.print(f"[dim]  2. Run: datasetpipeline run {file}[/dim]")


def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()