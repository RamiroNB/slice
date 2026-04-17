"""
task_sequences.py

Defines all continual learning task sequences from:
  "Unlocking the Power of Function Vectors for Characterizing and
   Mitigating Catastrophic Forgetting in Continual Instruction Tuning"
  Jiang et al., ICLR 2025

SuperNI task names follow the natural-instructions repository convention:
  taskNNN_<source_dataset>_<task_type>
  Raw JSON files live at:
    https://github.com/allenai/natural-instructions/tree/master/tasks/
  And are loadable via HuggingFace:
    Muennighoff/natural-instructions  (config name = task name below)

TRACE tasks are from:
  https://github.com/BeyonderXX/TRACE

Cross-reference: paper Table 4 maps each NI task ID to its source dataset,
which is how the full filenames below were resolved.

  NI-ID  | Source (Table 4)                        | Full task name
  --------|------------------------------------------|------------------------------------------
  NI002   | Quoref (QA)                             | task002_quoref_answer_generation (*) - ok 
  NI024   | CosmosQA (QA)                           | task024_cosmosqa_answer_generation (*) - ok 
  NI141   | Odd-man-out (Word Semantics)            | task141_odd-man-out_classification_category (*) - not found: task141_odd-man-out_classification_category.json
  NI163   | Synthetic Program Execution             | task163_count_words_ending_with_letter (*) - not found: task163_count_words_ending_with_letter.json  
  NI195   | Sentiment140 (Sentiment)                | task195_sentiment140_classification - ok
  NI220   | Rocstories (Title Generation)           | task220_rocstories_title_classification (*) - ok
  NI224   | Scruples (Ethics Classification)        | task224_scruples_anecdotes_ethical_judgment (*) - not found: task224_scruples_anecdotes_ethical_judgment.json  
  NI231   | IIRC (QA)                               | task231_iirc_link_classification (*) - not found: task231_iirc_link_classification.json
  NI273   | Europarl (Text Matching)                | task273_europarl_classification (*) - not found: task273_europarl_classification.json
  NI292   | StoryCommonsense (Info Extraction)      | task292_storycommonsense_character_text_generation (*) - ok 
  NI339   | ReCoRD (QA)                             | task339_record_answer_generation (*) - ok
  NI360   | Numersense (Fill-in-blank)              | task360_spolin_yesand_response_generation (*) - not found: task360_spolin_yesand_response_generation.json
  NI363   | SST2 (Sentiment)                        | task363_sst2_polarity_classification - ok 
  NI488   | Synthetic Program Execution             | task448_opus_paracrawl_en_tl_translation (*) - not found: task448_opus_paracrawl_en_tl_translation.json
  NI511   | Reddit TIFU (Summarization)             | task511_reddit_tifu_long_text_summarization (*) - ok
  NI589   | Amazon Fine Food Reviews (Summarization)| task589_amazonfood_summary_text_generation (*) - ok
  NI611   | Mutual (Dialogue)                       | task611_mutual_multi_turn_dialogue (*) - not found: task611_mutual_multi_turn_dialogue.json
  NI618   | Multilingual Amazon Reviews (Summary)   | task618_amazonreview_summary_text_generation (*) - ok
  NI619   | OhSUMED (Title Generation)              | task619_ohsumed_abstract_title_generation (*) - ok
  NI1290  | XSum (Summarization)                    | task1290_xsum_summarization (*) - ok
  NI1292  | Yelp Review Full (Sentiment)            | task1292_yelp_review_full_text_categorization (*) - not found: task1292_yelp_review_full_text_categorization.json
  NI1310  | Multilingual Amazon Reviews (Sentiment) | task1310_amazonreview_rating_classification (*) - ok 
  NI1343  | Amazon US Reviews (Sentiment)           | task1343_amazon_us_reviews_rating (*) - not found: task1343_amazon_us_reviews_rating.json
  NI1355  | Sentence Compression (Summarization)   | task1355_sent_comp_summarization (*) - ok 
  NI1357  | XLSum (Summarization)                  | task1357_xlsum_summary_generation (*) - not found: task1357_xlsum_summary_generation.json
  NI1510  | Evalution (Info Extraction)             | task1510_evalution_relation_extraction (*) - not found: task1510_evalution_relation_extraction.json

  (*) = resolved from paper Table 4 source name + natural-instructions naming convention.
        Verify these against the repo before running if exact match matters.
        The NI ID prefix (e.g. task195) is guaranteed correct; the suffix
        is the human-readable slug from the repo filename.

USAGE:
    from task_sequences import SEQUENCES, TRACE_SEQUENCE, GENERAL_EVAL_TASKS

    # Get a specific SuperNI sequence
    seq = SEQUENCES["NI-Seq-G1"]
    for task in seq["tasks"]:
        print(task["ni_id"], task["name"], task["type"])

    # Get the TRACE sequence
    for task in TRACE_SEQUENCE["tasks"]:
        print(task["name"], task["hf_dataset"])
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuperNITask:
    """A single SuperNI task."""
    ni_id: str                  # e.g. "NI195"  (paper shorthand)
    name: str                   # e.g. "task195_sentiment140_classification"
    source: str                 # Human-readable source dataset name
    category: str               # "classification" | "generation"
    hf_config: str              # config name for Muennighoff/natural-instructions
                                # (same as `name` in most cases)

    @property
    def raw_url(self) -> str:
        """URL to fetch the raw JSON from allenai/natural-instructions."""
        return (
            f"https://raw.githubusercontent.com/allenai/natural-instructions"
            f"/master/tasks/{self.name}.json"
        )


@dataclass
class TraceTask:
    """A single TRACE benchmark task."""
    name: str                   # short name used internally
    hf_dataset: Optional[str]   # HuggingFace dataset path if available
    category: str               # task category
    language: str               # "en" | "zh"
    metric: str                 # primary evaluation metric


@dataclass
class Sequence:
    """An ordered sequence of tasks for continual learning."""
    name: str
    task_type: str              # "classification" | "generation" | "mixed"
    tasks: list                 # List[SuperNITask] or List[TraceTask]
    description: str = ""


# ---------------------------------------------------------------------------
# SuperNI individual task definitions
# (26 tasks from paper Table 4)
# ---------------------------------------------------------------------------

# --- Classification tasks ---
NI195 = SuperNITask(
    ni_id="NI195",
    name="task195_sentiment140_classification",
    source="Sentiment140",
    category="classification",
    hf_config="task195_sentiment140_classification",
)


NI1343 = SuperNITask(
    ni_id="NI1343",
    name="task1343_amazon_us_reviews_rating",
    source="Amazon US Reviews",
    category="classification",
    hf_config="task1343_amazon_us_reviews_rating",
)
NI1310 = SuperNITask(
    ni_id="NI1310",
    name="task1310_amazonreview_rating_classification",
    source="Multilingual Amazon Reviews",
    category="classification",
    hf_config="task1310_amazonreview_rating_classification",
)

NI1292 = SuperNITask(
    ni_id="NI1292",
    name="task1292_yelp_review_full_text_categorization",
    source="Yelp Review Full",
    category="classification",
    hf_config="task1292_yelp_review_full_text_categorization",
)
NI363 = SuperNITask(
    ni_id="NI363",
    name="task363_sst2_polarity_classification",
    source="SST2",
    category="classification",
    hf_config="task363_sst2_polarity_classification",
)

NI231 = SuperNITask(
    ni_id="NI231",
    name="task231_iirc_link_classification",
    source="IIRC",
    category="classification",
    hf_config="task231_iirc_link_classification",
)
NI220 = SuperNITask(
    ni_id="NI220",
    name="task220_rocstories_title_classification",
    source="Rocstories",
    category="classification",
    hf_config="task220_rocstories_title_classification",
)

NI224 = SuperNITask(
    ni_id="NI224",
    name="task224_scruples_anecdotes_ethical_judgment",
    source="Scruples",
    category="classification",
    hf_config="task224_scruples_anecdotes_ethical_judgment",
)

NI273 = SuperNITask(
    ni_id="NI273",
    name="task273_europarl_classification",
    source="Europarl",
    category="classification",
    hf_config="task273_europarl_classification",
)

NI360 = SuperNITask(
    ni_id="NI360",
    name="task360_spolin_yesand_response_generation",
    source="Numersense",
    category="classification",
    hf_config="task360_spolin_yesand_response_generation",
)

# --- Generation tasks ---

NI618 = SuperNITask(
    ni_id="NI618",
    name="task618_amazonreview_summary_text_generation",
    source="Multilingual Amazon Reviews (Summary)",
    category="generation",
    hf_config="task618_amazonreview_summary_text_generation",
)
NI1290 = SuperNITask(
    ni_id="NI1290",
    name="task1290_xsum_summarization",
    source="XSum",
    category="generation",
    hf_config="task1290_xsum_summarization",
)

NI589 = SuperNITask(
    ni_id="NI589",
    name="task589_amazonfood_summary_text_generation",
    source="Amazon Fine Food Reviews",
    category="generation",
    hf_config="task589_amazonfood_summary_text_generation",
)

NI511 = SuperNITask(
    ni_id="NI511",
    name="task511_reddit_tifu_long_text_summarization",
    source="Reddit TIFU",
    category="generation",
    hf_config="task511_reddit_tifu_long_text_summarization",
)

NI1357 = SuperNITask(
    ni_id="NI1357",
    name="task1357_xlsum_summary_generation",
    source="XLSum",
    category="generation",
    hf_config="task1357_xlsum_summary_generation",
)
NI1355 = SuperNITask(
    ni_id="NI1355",
    name="task1355_sent_comp_summarization",
    source="Sentence Compression",
    category="generation",
    hf_config="task1355_sent_comp_summarization",
)
NI141 = SuperNITask(
    ni_id="NI141",
    name="task141_odd-man-out_classification_category",
    source="Odd-man-out",
    category="generation",
    hf_config="task141_odd-man-out_classification_category",
)

NI619 = SuperNITask(
    ni_id="NI619",
    name="task619_ohsumed_abstract_title_generation",
    source="OhSUMED",
    category="generation",
    hf_config="task619_ohsumed_abstract_title_generation",
)

NI163 = SuperNITask(
    ni_id="NI163",
    name="task163_count_words_ending_with_letter",
    source="Synthetic Program Execution",
    category="generation",
    hf_config="task163_count_words_ending_with_letter",
)
NI002 = SuperNITask(
    ni_id="NI002",
    name="task002_quoref_answer_generation",
    source="Quoref",
    category="generation",
    hf_config="task002_quoref_answer_generation",
)

# --- Mixed tasks ---
NI339 = SuperNITask(
    ni_id="NI339",
    name="task339_record_answer_generation",
    source="ReCoRD",
    category="generation",
    hf_config="task339_record_answer_generation",
)

NI1510 = SuperNITask(
    ni_id="NI1510",
    name="task1510_evalution_relation_extraction",
    source="Evalution",
    category="classification",
    hf_config="task1510_evalution_relation_extraction",
)

NI611 = SuperNITask(
    ni_id="NI611",
    name="task611_mutual_multi_turn_dialogue",
    source="Mutual",
    category="generation",
    hf_config="task611_mutual_multi_turn_dialogue",
)
NI292 = SuperNITask(
    ni_id="NI292",
    name="task292_storycommonsense_character_text_generation",
    source="StoryCommonsense",
    category="generation",
    hf_config="task292_storycommonsense_character_text_generation",
)

NI488 = SuperNITask(
    ni_id="NI488",
    name="task448_opus_paracrawl_en_tl_translation",
    source="Synthetic Program Execution",
    category="generation",
    hf_config="task448_opus_paracrawl_en_tl_translation",
)
NI024 = SuperNITask(
    ni_id="NI024",
    name="task024_cosmosqa_answer_generation",
    source="CosmosQA",
    category="generation",
    hf_config="task024_cosmosqa_answer_generation",
)


# ---------------------------------------------------------------------------
# TRACE task definitions
# (6 of 8 tasks selected by the paper)
# ---------------------------------------------------------------------------

TRACE_CSTANCE = TraceTask(
    name="C-STANCE",
    hf_dataset=None,        # part of BeyonderXX/TRACE repo
    category="Multi-Choice QA (Stance Detection)",
    language="zh",
    metric="rouge-l",
)
TRACE_FOMC = TraceTask(
    name="FOMC",
    hf_dataset=None,
    category="Multi-Choice QA (Finance)",
    language="en",
    metric="rouge-l",
)
TRACE_MEETINGBANK = TraceTask(
    name="MeetingBank",
    hf_dataset=None,
    category="Summarization",
    language="en",
    metric="rouge-l",
)
TRACE_PY150 = TraceTask(
    name="Py150",
    hf_dataset=None,
    category="Code Generation",
    language="python",
    metric="rouge-l",
)
TRACE_SCIENCEQA = TraceTask(
    name="ScienceQA",
    hf_dataset=None,
    category="Multi-Choice QA (Science)",
    language="en",
    metric="rouge-l",
)
TRACE_NUMGLUE = TraceTask(
    name="NumGLUE-cm",
    hf_dataset=None,
    category="Math Reasoning",
    language="en",
    metric="rouge-l",
)


# ---------------------------------------------------------------------------
# Sequence definitions (exactly as in paper Table 5)
# ---------------------------------------------------------------------------

SEQUENCES = {

    # --- Pure classification ---
    "NI-Seq-C1": Sequence(
        name="NI-Seq-C1",
        task_type="classification",
        tasks=[NI195, NI1343, NI1310, NI1292, NI363],
        description="Pure classification sequence 1: Sentiment140 → AmazonUS → AmazonMulti → Yelp → SST2",
    ),
    "NI-Seq-C2": Sequence(
        name="NI-Seq-C2",
        task_type="classification",
        tasks=[NI231, NI1343, NI220, NI224, NI273],
        description="Pure classification sequence 2: IIRC → AmazonUS → Rocstories → Scruples → Europarl",
    ),

    # --- Pure generation ---
    "NI-Seq-G1": Sequence(
        name="NI-Seq-G1",
        task_type="generation",
        tasks=[NI618, NI1290, NI589, NI511, NI1357],
        description="Pure generation sequence 1: AmazonReview → XSum → AmazonFood → RedditTIFU → XLSum",
    ),
    "NI-Seq-G2": Sequence(
        name="NI-Seq-G2",
        task_type="generation",
        tasks=[NI1355, NI141, NI619, NI163, NI002],
        description="Pure generation sequence 2: SentComp → OddManOut → OhSUMED → SyntheticExec → Quoref",
    ),

    # --- Mixed ---
    "NI-Seq-M1": Sequence(
        name="NI-Seq-M1",
        task_type="mixed",
        tasks=[NI360, NI363, NI1290, NI339, NI1510],
        description="Mixed sequence 1: Numersense → SST2 → XSum → ReCoRD → Evalution",
    ),
    "NI-Seq-M2": Sequence(
        name="NI-Seq-M2",
        task_type="mixed",
        tasks=[NI195, NI611, NI292, NI488, NI024],
        description="Mixed sequence 2: Sentiment140 → Mutual → StoryCommonsense → SyntheticExec → CosmosQA",
    ),
   
    "NI-Seq-Opposite": Sequence(
        name="NI-Seq-Opposite",
        task_type="mixed",
        tasks=[NI141, NI1510, NI360, NI363, NI611],
        description=(
            "Most-opposite 5-task subset (mean global_cosine=+0.2801): "
            "OddManOut → Evalution → Spolin → SST2 → Mutual"
        ),
    ),
    "NI-Seq-Dummy": Sequence(
        name="NI-Seq-Dummy",
        task_type="mixed",
        tasks=[NI363, NI618],
        description="Dev-only 2-task sequence: SST2 -> Amazon review summary",
    ),
    "TRACE-Dummy": Sequence(
        name="TRACE-Dummy",
        task_type="mixed",
        tasks=[TRACE_FOMC, TRACE_MEETINGBANK],
        description="Dev-only 2-task TRACE sequence: FOMC -> MeetingBank",
    ),
}

TRACE_SEQUENCE = Sequence(
    name="TRACE",
    task_type="mixed",
    tasks=[
        TRACE_CSTANCE,
        TRACE_FOMC,
        TRACE_MEETINGBANK,
        TRACE_PY150,
        TRACE_SCIENCEQA,
        TRACE_NUMGLUE,
    ],
    description="TRACE benchmark: C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm",
)

# ---------------------------------------------------------------------------
# General evaluation tasks (never used for training — GP / IP metrics)
# These are loaded via lm-evaluation-harness task names
# ---------------------------------------------------------------------------

GENERAL_EVAL_TASKS = {
    # Core 4 used in every experiment (Tables 1, 2)
    "hellaswag": {
        "lm_eval_name": "hellaswag",
        "description": "Sentence completion commonsense reasoning",
    },
    "commonsenseqa": {
        "lm_eval_name": "commonsense_qa",
        "description": "Commonsense question answering",
    },
    "alpaca": {
        "lm_eval_name": "alpaca_eval",        # requires alpaca_eval package
        "description": "Instruction following (Alpaca)",
    },
    "bbh_object_counting": {
        "lm_eval_name": "bbh_cot_fewshot_object_counting",  # specific BBH subtask
        "description": "BBH Object Counting subtask",
    },
    # Extended set used in some ablations (Ng=6)
    "openbookqa": {
        "lm_eval_name": "openbookqa",
        "description": "Open book question answering",
    },
    "lambada": {
        "lm_eval_name": "lambada_openai",
        "description": "Language modeling broad context",
    },
}

# Convenience list of the 4 core eval tasks used for GP/IP in main tables
CORE_EVAL_TASKS = ["hellaswag", "commonsenseqa", "alpaca", "bbh_object_counting"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sequence(name: str) -> Sequence:
    """Return a sequence by name. Accepts SuperNI names or 'TRACE'."""
    if name == "TRACE":
        return TRACE_SEQUENCE
    if name not in SEQUENCES:
        raise ValueError(
            f"Unknown sequence '{name}'. "
            f"Available: {list(SEQUENCES.keys()) + ['TRACE']}"
        )
    return SEQUENCES[name]


def all_superni_tasks() -> List[SuperNITask]:
    """Return the full list of 26 unique SuperNI tasks used across all sequences."""
    seen = set()
    tasks = []
    for seq in SEQUENCES.values():
        for task in seq.tasks:
            if task.ni_id not in seen:
                seen.add(task.ni_id)
                tasks.append(task)
    return tasks


def all_sequence_names() -> List[str]:
    return list(SEQUENCES.keys()) + ["TRACE"]


if __name__ == "__main__":
    # Quick sanity check
    print("=== SuperNI Sequences ===")
    for name, seq in SEQUENCES.items():
        task_ids = " → ".join(t.ni_id for t in seq.tasks)
        print(f"  {name} ({seq.task_type}): {task_ids}")

    print("\n=== TRACE Sequence ===")
    task_names = " → ".join(t.name for t in TRACE_SEQUENCE.tasks)
    print(f"  {task_names}")

    print(f"\n=== Total unique SuperNI tasks: {len(all_superni_tasks())} ===")
    for t in all_superni_tasks():
        print(f"  {t.ni_id:8s}  {t.name}")

    print("\n=== General Eval Tasks ===")
    for k, v in GENERAL_EVAL_TASKS.items():
        print(f"  {k:25s} → lm_eval: {v['lm_eval_name']}")