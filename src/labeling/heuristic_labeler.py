"""
Heuristic (rule-based) span labeler for FRLM router labels.

Classifies *obvious* entity mentions as **factual** or **linguistic**
without any API call, using lightweight lexical / pattern rules.
Texts that are ambiguous are returned as ``None`` so the caller can
fall back to the LLM labeler.

This module is designed to work alongside :class:`LLMLabeler` in a
hybrid pipeline:  heuristic first → batch-LLM for the remainder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.labeling.llm_labeler import SpanLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern sets
# ---------------------------------------------------------------------------

# Discourse markers / function words → almost always "linguistic"
_LINGUISTIC_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^(however|moreover|furthermore|therefore|thus|hence|"
               r"consequently|nevertheless|nonetheless|alternatively|"
               r"additionally|conversely|similarly|accordingly|"
               r"in\s+contrast|in\s+addition|on\s+the\s+other\s+hand|"
               r"taken\s+together|in\s+summary|in\s+conclusion|"
               r"for\s+example|for\s+instance|such\s+as|"
               r"these\s+results|this\s+study|our\s+findings|"
               r"we\s+found|we\s+observed|we\s+demonstrated|"
               r"it\s+is|it\s+was|there\s+is|there\s+are|there\s+was|"
               r"has\s+been|have\s+been|were\s+found|"
               r"further\s+investigation|further\s+research|"
               r"the\s+present\s+study|the\s+current\s+study|"
               r"as\s+shown|as\s+described|as\s+reported|"
               r"not\s+significant|no\s+significant|significantly|"
               r"respectively|approximately|overall|previously|"
               r"recently|generally|typically|usually|often|"
               r"detected|examined|evaluated|assessed|analyzed|analysed|"
               r"performed|conducted|compared|included|obtained|"
               r"reported|described|demonstrated|determined|confirmed|"
               r"identified|investigated|suggested|indicated|observed|"
               r"measured|calculated|considered|presented|discussed|"
               r"patients|subjects|participants|controls|samples|"
               r"methods?|results?|conclusions?|discussion|background|"
               r"introduction|objectives?|purpose|aims?)$",
               re.IGNORECASE),
]

# Strong biomedical entity signals → almost always "factual"
_FACTUAL_PATTERNS: List[re.Pattern[str]] = [
    # Drug-like names (ends in -ib, -ab, -mab, -nib, -ol, -in, -ide, -ine, etc.)
    re.compile(r"(?i)\b\w+(?:inib|tinib|ciclib|rafenib|zumab|ximab|"
               r"mumab|lizumab|tuzumab|cillin|mycin|taxel|platin|"
               r"mustine|rubicin|clovir|navir|asone|olone|"
               r"prednisone|methasone|olol|pril|sartan)\b"),
    # Gene/protein symbols: 2-6 uppercase letters/digits (e.g., EGFR, TP53, BRCA1)
    re.compile(r"^[A-Z][A-Z0-9]{1,5}$"),
    # Measurements: numbers with units
    re.compile(r"\d+\s*(?:mg|μg|ng|ml|μl|mM|μM|nM|kg|g/[dml]|"
               r"IU|%|mmol|nmol|pmol|cm|mm|μm|kDa|Da|Hz|"
               r"Gy|cGy|MBq|mCi)\b"),
    # IC50, EC50, Ki, Kd, etc.
    re.compile(r"(?i)\b(?:IC50|EC50|ED50|LD50|Ki|Kd|Ka|Km|Vmax|"
               r"AUC|Cmax|Tmax|t1/2|half-life)\b"),
    # Specific pathway / receptor names
    re.compile(r"(?i)\b(?:p53|p21|p27|Rb|NF-κB|NF-kB|"
               r"MAPK|ERK|JNK|PI3K|AKT|mTOR|VEGF|HER2|"
               r"BCR-ABL|ALK|ROS1|BRAF|KRAS|NRAS|MET|RET|"
               r"PD-L1|PD-1|CTLA-4|TNF-α|TNF-alpha|IL-\d+|"
               r"TGF-β|TGF-beta|IGF-\d+|EGF|FGF|PDGF|"
               r"JAK\d?|STAT\d?|Wnt|Hedgehog|Notch)\b"),
    # Cancer types
    re.compile(r"(?i)\b(?:carcinoma|sarcoma|lymphoma|leukemia|leukaemia|"
               r"melanoma|glioma|glioblastoma|adenocarcinoma|"
               r"mesothelioma|neuroblastoma|retinoblastoma|"
               r"hepatocellular|cholangiocarcinoma|"
               r"NSCLC|SCLC|CLL|AML|ALL|CML|DLBCL|"
               r"triple[- ]negative|estrogen[- ]receptor|"
               r"prostate\s+cancer|breast\s+cancer|lung\s+cancer|"
               r"colorectal\s+cancer|pancreatic\s+cancer|"
               r"ovarian\s+cancer|gastric\s+cancer)\b"),
    # Chromosomal locations
    re.compile(r"(?i)\b(?:chromosome\s+)?\d{1,2}[pq]\d"),
    # CUI / UMLS-style identifiers
    re.compile(r"^C\d{7}$"),
    # Medical imaging / diagnostic terms
    re.compile(r"(?i)\b(?:mammography|sonography|ultrasonography|"
               r"radiography|tomography|fluoroscopy|angiography|"
               r"echocardiography|endoscopy|colonoscopy|bronchoscopy|"
               r"MRI|MR|CT|PET|SPECT|fMRI|X-ray|ultrasound)\b"),
    # Common biomedical procedure / anatomy / physiology terms
    re.compile(r"(?i)\b(?:biopsy|resection|ablation|excision|"
               r"mastectomy|lumpectomy|lobectomy|nephrectomy|"
               r"chemotherapy|radiotherapy|immunotherapy|"
               r"apoptosis|necrosis|angiogenesis|metastasis|"
               r"proliferation|differentiation|phosphorylation|"
               r"methylation|transcription|translation|"
               r"sensitivity|specificity|prevalence|incidence|"
               r"survival|mortality|morbidity|prognosis|"
               r"biomarker|antibody|antigen|enzyme|receptor|"
               r"ligand|inhibitor|agonist|antagonist|"
               r"cytokine|chemokine|interferon|interleukin|"
               r"serum|plasma|tissue|tumor|tumour|lesion|"
               r"nodule|polyp|cyst|fibrosis|neoplasm|"
               r"adenoma|papilloma|dysplasia|hyperplasia)\b"),
    # Multi-word terms with medical modifiers (e.g., "invasive cancers",
    # "benign tumours", "lobular carcinoma", "MR mammography")
    re.compile(r"(?i)\b(?:invasive|non-invasive|noninvasive|benign|"
               r"malignant|metastatic|recurrent|primary|secondary|"
               r"adjuvant|neoadjuvant|palliative|curative|"
               r"histological|pathological|clinical|diagnostic|"
               r"intraductal|lobular|ductal|papillary|mucinous|"
               r"squamous|basal|medullary|tubular|serous|"
               r"microinvasive|multicentric|multifocal|bilateral|"
               r"unilateral|contralateral|ipsilateral)\s+\w+"),
    # Staining / lab techniques
    re.compile(r"(?i)\b(?:immunohistochemistry|IHC|FISH|PCR|"
               r"RT-PCR|qPCR|ELISA|Western\s+blot|"
               r"flow\s+cytometry|mass\s+spectrometry|"
               r"electrophoresis|chromatography|"
               r"hematoxylin|eosin|H&E|Ki-67|Ki67)\b"),
    # Anatomical / organ terms
    re.compile(r"(?i)\b(?:breast|lung|liver|kidney|colon|"
               r"pancreas|ovary|ovarian|prostate|thyroid|"
               r"brain|spine|spinal|bone|lymph\s+node|"
               r"axillary|mediastinal|peritoneal|pleural|"
               r"hepatic|renal|pulmonary|cardiac|cerebral|"
               r"gastrointestinal|genitourinary)\b"),
    # Clinical staging / grading
    re.compile(r"(?i)\b(?:stage\s+[I]{1,3}V?|grade\s+\d|"
               r"T[0-4]N[0-3]M[01]|TNM|FIGO|Gleason|"
               r"RECIST|WHO\s+grade|ECOG|Karnofsky)\b"),
]


# ---------------------------------------------------------------------------
# HeuristicLabeler
# ---------------------------------------------------------------------------


class HeuristicLabeler:
    """Classify obvious text spans without an API call.

    For each text, returns a ``SpanLabel`` if the classification is
    high-confidence, or ``None`` if the text should be sent to the LLM.

    Parameters
    ----------
    factual_confidence : float
        Confidence to assign heuristic factual labels.
    linguistic_confidence : float
        Confidence to assign heuristic linguistic labels.
    max_text_length : int
        Texts longer than this are always sent to the LLM.
    """

    def __init__(
        self,
        factual_confidence: float = 0.88,
        linguistic_confidence: float = 0.85,
        max_text_length: int = 80,
    ) -> None:
        self.factual_confidence = factual_confidence
        self.linguistic_confidence = linguistic_confidence
        self.max_text_length = max_text_length
        self._stats: Dict[str, int] = {
            "factual": 0,
            "linguistic": 0,
            "deferred": 0,
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def try_label(self, text: str) -> Optional[SpanLabel]:
        """Attempt to label *text* heuristically.

        Returns
        -------
        SpanLabel or None
            A label if the text is obviously factual or linguistic,
            ``None`` if it should be deferred to the LLM.
        """
        stripped = text.strip()
        if not stripped:
            return None

        # Long texts are always deferred — need contextual judgement
        if len(stripped) > self.max_text_length:
            self._stats["deferred"] += 1
            return None

        # ---- Check linguistic patterns first (common function words) ----
        for pat in _LINGUISTIC_PATTERNS:
            if pat.fullmatch(stripped):
                self._stats["linguistic"] += 1
                return SpanLabel(
                    start_char=0,
                    end_char=len(text),
                    text=text,
                    label="linguistic",
                    confidence=self.linguistic_confidence,
                )

        # ---- Check factual patterns (biomedical entities) ----
        for pat in _FACTUAL_PATTERNS:
            if pat.search(stripped):
                self._stats["factual"] += 1
                return SpanLabel(
                    start_char=0,
                    end_char=len(text),
                    text=text,
                    label="factual",
                    confidence=self.factual_confidence,
                )

        # Ambiguous — defer to LLM
        self._stats["deferred"] += 1
        return None

    # ------------------------------------------------------------------
    # Bulk classification
    # ------------------------------------------------------------------

    def classify_batch(
        self, texts: List[str]
    ) -> List[Optional[SpanLabel]]:
        """Label a batch, returning ``None`` for texts that need the LLM."""
        return [self.try_label(t) for t in texts]

    @property
    def stats(self) -> Dict[str, int]:
        """Cumulative counts of factual / linguistic / deferred texts."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {"factual": 0, "linguistic": 0, "deferred": 0}


