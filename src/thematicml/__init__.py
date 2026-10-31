"""ThematicML — local-first machine learning for qualitative thematic analysis.

Classical NLP, clustering, topic models, graph algorithms and time-series
statistics. No LLMs or generative AI services are used.
"""
__version__ = "0.1.0"

from thematicml.config import AnalysisConfig, CodebookRule, ColumnMapping  # noqa: E402
from thematicml.storage.projects import AnalysisResult, Metrics, Project  # noqa: E402

__all__ = ["AnalysisConfig", "AnalysisResult", "CodebookRule", "ColumnMapping", "Metrics", "Project", "__version__"]
