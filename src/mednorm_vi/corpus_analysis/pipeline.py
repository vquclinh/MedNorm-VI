"""Phase 2A pipeline: public corpus -> existing L1 -> descriptive statistics.

Every document is parsed through the EXISTING L1 pipeline with its existing
config — Phase 2A changes no preprocessing behavior and adds no new parsing. It
then measures structure and layout only: no prediction, no entity extraction, no
linking, no LLM, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..case_router.signals import RouterConfig, load_router_config
from ..document_intelligence import analyze_text
from ..document_intelligence.builder import load_config as load_l1_config
from ..document_intelligence.models import L1Config
from ..document_intelligence.sections import SectionLexicon
from ..document_intelligence.validation import validate_graph
from .config import AnalysisConfig, load_analysis_config
from .corpus import aggregate
from .document import analyze_document
from .loader import CorpusDocument, corpus_sha256, load_corpus
from .models import CorpusAnalysis


@dataclass(frozen=True, slots=True)
class Phase2AConfig:
    """Resolved Phase 2A configuration (reuses the existing L1 + router configs)."""

    analysis: AnalysisConfig
    router: RouterConfig
    l1_config_hash: str
    l1_config: L1Config
    l1_lexicon: SectionLexicon

    @staticmethod
    def load(
        analysis_config: str | Path = "configs/corpus_analysis/analysis_v1.yaml",
        router_config: str | Path = "configs/case_router/base.yaml",
        l1_config: str | Path = "configs/document_intelligence/base.yaml",
    ) -> Phase2AConfig:
        cfg, lex = load_l1_config(l1_config)
        return Phase2AConfig(
            analysis=load_analysis_config(analysis_config),
            router=load_router_config(router_config),
            l1_config_hash=cfg.config_hash, l1_config=cfg, l1_lexicon=lex)


def run_corpus_analysis(
    documents: list[CorpusDocument], config: Phase2AConfig, *, corpus_id: str = "public_input",
) -> CorpusAnalysis:
    """Analyze an already-loaded corpus deterministically."""
    analyses = []
    for doc in documents:
        # EXISTING L1 pipeline, existing config — unchanged behavior. No
        # source_path: document_id stays content-derived and machine-independent.
        graph = analyze_text(doc.text, config=config.l1_config, lexicon=config.l1_lexicon)
        l1_valid = validate_graph(graph).ok
        analyses.append(analyze_document(
            doc, graph, router_config=config.router, config=config.analysis,
            l1_valid=l1_valid))
    return aggregate(analyses, corpus_id=corpus_id, corpus_sha256=corpus_sha256(documents),
                     config=config.analysis, l1_config_hash=config.l1_config_hash)


def analyze_corpus_dir(
    directory: str | Path, config: Phase2AConfig, *, pattern: str = "*.txt",
    corpus_id: str = "public_input",
) -> CorpusAnalysis:
    """Load a corpus directory (read-only) and analyze it."""
    return run_corpus_analysis(load_corpus(directory, pattern=pattern), config,
                               corpus_id=corpus_id)


__all__ = ["Phase2AConfig", "run_corpus_analysis", "analyze_corpus_dir"]
