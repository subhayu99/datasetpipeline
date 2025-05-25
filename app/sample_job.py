"""
Enhanced sample job configurations with different templates and complexity levels.

This module provides various pre-built configurations for common use cases,
making it easier for users to get started with DatasetPipeline.
"""

from enum import Enum
from .job import Job, JobConfig
from .loader import LoaderConfig, HFLoaderConfig
from .format import (
    FormatConfig,
    MergerFormatConfig,
    FieldConfig,
    SFTFormatConfig,
    Role,
    DPOFormatConfig,
    DPOColumns,
    ConversationalFormatConfig,
    ConversationalTextFormatConfig,
    ToTextFormatConfig,
    RoleConfig,
    OutputFormatConfig,
)
from .analyzer import AnalyzerConfig, QualityAnalyzerConfig
from .dedup import DedupConfig, SemanticDedupConfig
from .saver import SaverConfig, LocalSaverConfig, FileType


# Minimal configuration for quick start
minimal_config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="teknium/OpenHermes-2.5",
            split="train",
            take_rows=1000,
        ),
        local=None,
    ),
    format=FormatConfig(
        sft=SFTFormatConfig(
            use_openai=False,
        ),
        # All other format options set to None for clarity
        merger=None,
        dpo=None,
        conv=None,
        conv_text=None,
        to_text=None,
        output=None,
    ),
    deduplicate=None,  # Explicitly disabled
    analyze=None,      # Explicitly disabled
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="output",
            filetype=FileType.JSON,
            filename="processed-openhermes2.5",
        )
    ),
)

# SFT-focused template
sft_template_config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="teknium/OpenHermes-2.5",
            split="train",
            take_rows=10000,
        ),
        local=None,
    ),
    format=FormatConfig(
        sft=SFTFormatConfig(
            use_openai=True,
            column_role_map={
                "system": Role.SYSTEM,
                "human": Role.USER,
                "gpt": Role.ASSISTANT,
            },
        ),
        output=OutputFormatConfig(
            return_only_messages=True,
        ),
        # Unused format options
        merger=None,
        dpo=None,
        conv=None,
        conv_text=None,
        to_text=None,
    ),
    deduplicate=DedupConfig(
        semantic=SemanticDedupConfig(
            threshold=0.85,
            column="messages",
        )
    ),
    analyze=None,  # Skip analysis for faster processing
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="sft_training_data",
            filetype=FileType.JSON,
            filename="processed-openhermes2.5-sft",
        )
    ),
)

# DPO-focused template
dpo_template_config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="Anthropic/hh-rlhf",
            split="train",
            take_rows=5000,
        ),
        local=None,
    ),
    format=FormatConfig(
        dpo=DPOFormatConfig(
            column_role_map={
                "human": DPOColumns.USER,
                "system": DPOColumns.SYSTEM,
                "chosen": DPOColumns.CHOSEN,
                "rejected": DPOColumns.REJECTED,
            }
        ),
        output=OutputFormatConfig(
            return_only_messages=True,
        ),
        # Unused format options
        merger=None,
        sft=None,
        conv=None,
        conv_text=None,
        to_text=None,
    ),
    deduplicate=None,  # DPO often doesn't need deduplication
    analyze=None,      # Skip analysis for this template
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="dpo_training_data",
            filetype=FileType.JSON,
            filename="processed-hh-rlhf-dpo",
        )
    ),
)

# Quality analysis focused template
analysis_template_config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="your-dataset/name",  # User needs to change this
            split="train",
        ),
        local=None,
    ),
    format=FormatConfig(
        sft=SFTFormatConfig(
            use_openai=True,
            column_role_map={},
        ),
        # Other format options
        merger=None,
        dpo=None,
        conv=None,
        conv_text=None,
        to_text=None,
        output=None,
    ),
    deduplicate=None,  # Focus on analysis, skip dedup
    analyze=AnalyzerConfig(
        quality=QualityAnalyzerConfig(
            column_name="messages",
            categories=["code", "reasoning", "creative", "factual"],
        )
    ),
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="analyzed_data",
            filetype=FileType.PARQUET,
            filename="your-dataset-name-analyzed",
        )
    ),
)

# Full comprehensive configuration (your original)
comprehensive_config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="davanstrien/data-centric-ml-sft",
        ),
    ),
    format=FormatConfig(
        merger=MergerFormatConfig(
            user=FieldConfig(
                fields=["book_id", "author", "text"],
                separator="\n",
                merged_field="human",
            ),
        ),
        sft=SFTFormatConfig(
            use_openai=False,
            column_role_map={
                "persona": Role.SYSTEM,
                "human": Role.USER,
                "summary": Role.ASSISTANT,
            },
        ),
        dpo=DPOFormatConfig(
            column_role_map={
                "human": "user",  # we can pass string
                "persona": DPOColumns.SYSTEM,  # or DPOColumns
                "positive": DPOColumns.CHOSEN,
                "negative": DPOColumns.REJECTED,
            }
        ),
        conv=ConversationalFormatConfig(),
        conv_text=ConversationalTextFormatConfig(),
        to_text=ToTextFormatConfig(
            system=RoleConfig(
                template="SYSTEM: {system}",
                key="system",
            ),
            user=RoleConfig(
                template="USER: {user}",
                key="user",
            ),
            assistant=RoleConfig(
                template="ASSISTANT: {assistant}",
                key="assistant",
            ),
            separator="\n\n",
        ),
        output=OutputFormatConfig(
            return_only_messages=True,
        ),
    ),
    deduplicate=DedupConfig(
        semantic=SemanticDedupConfig(
            threshold=0.8,
        )
    ),
    analyze=AnalyzerConfig(
        quality=QualityAnalyzerConfig(
            column_name="messages",
            categories=[
                "code", "math", "job", "essay", "translation", "literature",
                "history", "science", "medicine", "news", "finance", "geography",
                "philosophy", "psychology", "education", "art", "music", "technology",
                "environment", "food", "sports", "fashion", "travel", "culture",
                "language", "religion", "politics", "space", "entertainment",
                "healthcare", "animals", "weather", "architecture", "automotive",
                "business", "comedy", "crime", "diy", "economics", "gaming",
                "law", "marketing", "parenting", "science_fiction", "social_media",
                "mythology", "folklore", "astrology", "horror", "mystery",
            ],
        )
    ),
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="processed",
            filetype=FileType.PARQUET,
            filename="processed-davanstrien-data-centric-ml-full",
        )
    ),
)


class TemplateType(str, Enum):
    MINIMAL = "minimal"
    SFT = "sft"
    DPO = "dpo"
    ANALYSIS = "analysis"
    FULL = "full"


# Mapping for easy access
TEMPLATE_CONFIGS = {
    TemplateType.MINIMAL: minimal_config,
    TemplateType.SFT: sft_template_config,
    TemplateType.DPO: dpo_template_config,
    TemplateType.ANALYSIS: analysis_template_config,
    TemplateType.FULL: comprehensive_config,
}

def get_config_by_type(template: TemplateType = TemplateType.MINIMAL) -> JobConfig:
    """Get a configuration by template type."""
    return TEMPLATE_CONFIGS.get(template, minimal_config)

# Keep the original for backward compatibility
config = comprehensive_config

if __name__ == "__main__":
    # Demo all configurations
    for name, cfg in TEMPLATE_CONFIGS.items():
        print(f"\n=== {name.upper()} CONFIGURATION ===")
        job = Job(cfg)
        print(job.to_yaml())
        print("=" * 50)