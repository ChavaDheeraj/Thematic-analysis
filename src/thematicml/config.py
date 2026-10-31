"""Analysis configuration.

Every parameter that influences a result lives here so that a run can be
reproduced exactly from its stored configuration.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CodebookRule(BaseModel):
    """A researcher-defined code matched by keywords or phrases.

    Keywords are matched case-insensitively on word boundaries. A trailing
    ``*`` acts as a prefix wildcard (``navigat*`` matches ``navigation``).
    """

    label: str
    keywords: list[str]
    description: str = ""
    category: str | None = None


class ColumnMapping(BaseModel):
    """Maps columns of an uploaded table to the fields ThematicML understands."""

    text_col: str
    id_col: str | None = None
    time_col: str | None = None
    group_col: str | None = None
    label_col: str | None = None
    metadata_cols: list[str] = Field(default_factory=list)
    duplicate_policy: Literal["flag", "exclude"] = "flag"
    # text features, stopwords and sentiment are English-only; other-language records can be set aside
    language_policy: Literal["keep", "exclude_non_english"] = "keep"


class GNNConfig(BaseModel):
    """Optional GraphSAGE experiment on the code co-occurrence graph (neural; needs thematicml[gnn])."""

    task: Literal["link_prediction", "node_classification"] = "link_prediction"
    # model
    hidden_dim: int = Field(64, ge=4, le=1024)
    out_dim: int = Field(32, ge=2, le=512)
    num_layers: int = Field(2, ge=1, le=6)
    dropout: float = Field(0.3, ge=0.0, le=0.9)
    decoder: Literal["dot", "mlp"] = "dot"
    feature_set: Literal["label+frequency", "label", "identity"] = "label+frequency"
    # optimisation
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    weight_decay: float = Field(5e-4, ge=0.0, le=1.0)
    epochs: int = Field(300, ge=1, le=5000)
    patience: int = Field(30, ge=1, le=1000)
    # evaluation
    val_fraction: float = Field(0.15, gt=0.0, lt=0.5)
    test_fraction: float = Field(0.15, gt=0.0, lt=0.5)
    # share of training edges held out of message passing and used only as targets (0 = PyG default)
    disjoint_train_ratio: float = Field(0.0, ge=0.0, lt=0.9)
    repeats: int = Field(3, ge=1, le=20)
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    top_k_predictions: int = Field(50, ge=0, le=1000)
    # graph construction overrides (default: the run's configuration)
    cooccurrence_level: Literal["record", "segment"] | None = None
    min_cooccurrence: int | None = Field(None, ge=1)
    min_npmi: float | None = Field(None, ge=-1.0, le=1.0)

    @field_validator("test_fraction")
    @classmethod
    def _fractions(cls, v: float, info):
        if v + info.data.get("val_fraction", 0.15) >= 0.8:
            raise ValueError("val_fraction + test_fraction must be below 0.8")
        return v


class AnalysisConfig(BaseModel):
    # --- column mapping (used by the Python API; the web app maps columns at upload) ---
    text_col: str | None = None
    id_col: str | None = None
    time_col: str | None = None
    group_col: str | None = None
    label_col: str | None = None

    # --- segmentation ---
    segmentation: Literal["record", "sentence", "paragraph"] = "sentence"
    min_segment_words: int = Field(3, ge=1)
    # analyse a passage once when the same respondent's same text (>= 8 words) arrives through several uploads,
    # e.g. a page or file saved twice
    dedupe_across_datasets: bool = True

    # --- preprocessing / features ---
    lowercase: bool = True
    remove_stopwords: bool = True
    keep_negations: bool = True
    custom_stopwords: list[str] = Field(default_factory=list)
    synonyms: dict[str, str] = Field(default_factory=dict)
    ngram_min: int = Field(1, ge=1, le=3)
    ngram_max: int = Field(2, ge=1, le=4)
    min_df: int = Field(2, ge=1)
    max_df: float = Field(0.9, gt=0.0, le=1.0)
    max_features: int = Field(5000, ge=50)

    # --- initial coding ---
    coding_method: Literal["tfidf", "rules", "both"] = "tfidf"
    n_codes: int = Field(40, ge=1, le=500)
    min_code_support: int = Field(3, ge=1)
    phrase_boost: float = Field(1.25, ge=1.0, le=3.0)
    codebook: list[CodebookRule] = Field(default_factory=list)
    # with coding_method "both", skip TF-IDF terms a codebook keyword already matches so inductive codes add new ground
    complement_codebook: bool = True
    supervised: bool = False

    # --- clustering ---
    clustering_method: Literal["kmeans", "hierarchical"] = "kmeans"
    n_clusters: int | None = Field(None, ge=2, le=100)
    k_min: int = Field(2, ge=2)
    k_max: int = Field(12, ge=2, le=100)
    svd_components: int = Field(100, ge=2, le=500)
    max_hierarchical_segments: int = Field(8000, ge=100)

    # --- topics ---
    topic_method: Literal["nmf", "lda"] = "nmf"
    n_topics: int = Field(8, ge=2, le=100)
    n_top_terms: int = Field(10, ge=3, le=30)
    topic_min_weight: float = Field(0.3, ge=0.0, le=1.0)

    # --- graph ---
    graph_method: Literal["louvain", "leiden", "greedy", "label_propagation"] = "louvain"
    cooccurrence_level: Literal["record", "segment"] = "record"
    min_cooccurrence: int = Field(2, ge=1)
    # keep co-occurrence edges only when codes co-occur more than expected by chance (NPMI > min_npmi)
    min_npmi: float = Field(0.0, ge=-1.0, le=1.0)
    similarity_edges: bool = True
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    resolution: float = Field(1.0, gt=0.0, le=10.0)
    link_prediction_baseline: bool = True

    # --- themes ---
    min_theme_codes: int = Field(2, ge=1)
    theme_label_template: str = "{1} & {2}"
    n_representative: int = Field(5, ge=1, le=50)

    # --- sentiment ---
    sentiment: bool = True

    # --- temporal ---
    temporal: bool = True
    temporal_freq: Literal["auto", "D", "W", "M", "Q"] = "auto"
    n_graph_windows: int = Field(4, ge=2, le=24)
    temporal_top_codes: int = Field(10, ge=1, le=50)

    random_state: int = 42

    # --- optional GNN experiment, queued after the pipeline completes (None = off) ---
    gnn: GNNConfig | None = None

    @field_validator("ngram_max")
    @classmethod
    def _ngram_order(cls, v: int, info):
        lo = info.data.get("ngram_min", 1)
        if v < lo:
            raise ValueError("ngram_max must be >= ngram_min")
        return v

    @field_validator("k_max")
    @classmethod
    def _k_order(cls, v: int, info):
        lo = info.data.get("k_min", 2)
        if v < lo:
            raise ValueError("k_max must be >= k_min")
        return v

    def mapping(self) -> ColumnMapping:
        if not self.text_col:
            raise ValueError("text_col is required to map a dataset")
        return ColumnMapping(
            text_col=self.text_col,
            id_col=self.id_col,
            time_col=self.time_col,
            group_col=self.group_col,
            label_col=self.label_col,
        )
