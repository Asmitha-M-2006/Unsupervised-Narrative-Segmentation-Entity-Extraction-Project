from collections import Counter
from functools import lru_cache
import re

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load the SentenceTransformer model once and reuse it.

    Returns:
        SentenceTransformer: MiniLM sentence embedding model.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_nlp_model():
    """
    Load spaCy's English NLP pipeline once and reuse it.

    Returns:
        spacy.Language: English spaCy model.
    """
    return spacy.load("en_core_web_sm")


# ============================================================
# TRANSCRIPT EXTRACTION
# ============================================================

def extract_transcript(video_id):
    """
    Fetch the YouTube transcript for a video.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        list: Transcript snippets.
    """
    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    return transcript


# ============================================================
# TRANSCRIPT WINDOWING
# ============================================================

def create_windows(transcript, window_size=75):
    """
    Convert small ASR transcript snippets into fixed-size
    word windows.

    Args:
        transcript: Fetched transcript snippets.
        window_size (int): Number of words per semantic window.

    Returns:
        list[dict]: Windows containing text and timestamps.
    """

    windows = []

    current_words = []
    current_start = None
    current_end = None

    for snippet in transcript:

        words = snippet.text.split()

        if not words:
            continue

        if current_start is None:
            current_start = snippet.start

        current_words.extend(words)

        current_end = snippet.start + snippet.duration

        if len(current_words) >= window_size:

            windows.append(
                {
                    "id": len(windows),
                    "text": " ".join(current_words),
                    "start": current_start,
                    "end": current_end,
                }
            )

            current_words = []
            current_start = None
            current_end = None

    # Preserve final partial window.
    if current_words:

        windows.append(
            {
                "id": len(windows),
                "text": " ".join(current_words),
                "start": current_start,
                "end": current_end,
            }
        )

    return windows


# ============================================================
# SENTENCE EMBEDDINGS
# ============================================================

def create_embeddings(windows):
    """
    Convert semantic windows into dense vector representations.

    Args:
        windows (list[dict]): Transcript windows.

    Returns:
        np.ndarray: Embedding matrix.
    """

    model = load_embedding_model()

    texts = [
        window["text"]
        for window in windows
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


# ============================================================
# NARRATIVE SEGMENTATION
# ============================================================

def _calculate_consecutive_similarity(embeddings):
    """
    Calculate cosine similarity between consecutive windows.

    Args:
        embeddings (np.ndarray): Window embeddings.

    Returns:
        np.ndarray: Similarity between every consecutive pair.
    """

    if len(embeddings) < 2:
        return np.array([])

    similarities = cosine_similarity(
        embeddings[:-1],
        embeddings[1:]
    ).diagonal()

    return np.asarray(
        similarities,
        dtype=np.float32
    )


def _smooth_similarity(
    similarities,
    smoothing_radius=2
):
    """
    Smooth the consecutive similarity curve.

    This reduces isolated noisy similarity changes caused
    by ASR errors.

    Args:
        similarities (np.ndarray): Raw similarities.
        smoothing_radius (int): Smoothing radius.

    Returns:
        np.ndarray: Smoothed similarities.
    """

    if len(similarities) == 0:
        return similarities

    kernel_size = (
        smoothing_radius * 2
    ) + 1

    kernel = (
        np.ones(kernel_size, dtype=np.float32)
        / kernel_size
    )

    padded = np.pad(
        similarities,
        (
            smoothing_radius,
            smoothing_radius
        ),
        mode="edge"
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid"
    )


def _calculate_context_change(
    embeddings,
    radius=4
):
    """
    Measure semantic change around every possible boundary.

    Instead of comparing only:

        window i <-> window i+1

    we compare:

        average(previous context)
                    vs
        average(next context)

    This makes the score more representative of a narrative
    transition rather than a small local change.

    Args:
        embeddings (np.ndarray): Window embeddings.
        radius (int): Number of windows used on each side.

    Returns:
        np.ndarray: Context-change score for each boundary.
    """

    if len(embeddings) < (radius * 2 + 1):
        return np.zeros(
            max(len(embeddings) - 1, 0),
            dtype=np.float32
        )

    scores = np.zeros(
        len(embeddings) - 1,
        dtype=np.float32
    )

    for boundary in range(
        radius,
        len(embeddings) - radius
    ):

        left_context = embeddings[
            boundary - radius:boundary
        ]

        right_context = embeddings[
            boundary + 1:boundary + radius + 1
        ]

        left_mean = np.mean(
            left_context,
            axis=0,
            keepdims=True
        )

        right_mean = np.mean(
            right_context,
            axis=0,
            keepdims=True
        )

        similarity = cosine_similarity(
            left_mean,
            right_mean
        )[0][0]

        scores[boundary] = 1.0 - similarity

    return scores


def _find_candidate_boundaries(
    similarities,
    context_change,
    threshold,
    context_threshold,
    local_radius=2
):
    """
    Find candidate narrative boundaries.

    A candidate must satisfy two conditions:

    1. Consecutive similarity is locally low.
    2. The broader context on either side is sufficiently
       different.

    This prevents small changes inside the same story from
    being treated as narrative boundaries.
    """

    candidates = []

    for i in range(
        local_radius,
        len(similarities) - local_radius
    ):

        current = similarities[i]

        if current >= threshold:
            continue

        left = similarities[
            i - local_radius:i
        ]

        right = similarities[
            i + 1:i + local_radius + 1
        ]

        neighborhood = np.concatenate(
            [left, right]
        )

        if current > np.min(neighborhood):
            continue

        if context_change[i] < context_threshold:
            continue

        candidates.append(i)

    return candidates


def _select_boundaries(
    candidates,
    similarities,
    context_change,
    min_segment_windows
):
    """
    Select the strongest candidate boundaries.

    Candidates that occur close to each other are treated as
    belonging to the same transition region. Only the strongest
    boundary in that region is retained.

    Boundary strength combines:

        - semantic context change
        - local similarity drop
    """

    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda i: (
            -context_change[i],
            similarities[i]
        )
    )

    selected = []

    for candidate in ranked:

        if all(
            abs(candidate - existing)
            >= min_segment_windows
            for existing in selected
        ):
            selected.append(candidate)

    return sorted(selected)


def _add_max_length_boundaries(
    boundaries,
    similarities,
    total_windows,
    max_segment_windows
):
    """
    Prevent extremely long segments.

    If no semantic boundary occurs for too long, place a
    forced boundary at the weakest similarity point inside
    the allowed region.
    """

    boundaries = sorted(boundaries)

    result = []

    previous = -1

    for boundary in boundaries:

        while (
            boundary - previous
            > max_segment_windows
        ):

            start = previous + 1

            end = min(
                start + max_segment_windows,
                boundary
            )

            if end <= start:
                break

            local_range = similarities[
                start:end
            ]

            forced_offset = int(
                np.argmin(local_range)
            )

            forced_boundary = (
                start + forced_offset
            )

            if (
                result
                and forced_boundary <= result[-1]
            ):
                break

            result.append(
                forced_boundary
            )

            previous = forced_boundary

        result.append(boundary)

        previous = boundary

    # Handle final segment.
    while (
        total_windows - 1 - previous
        > max_segment_windows
    ):

        start = previous + 1

        end = min(
            start + max_segment_windows,
            total_windows - 1
        )

        if end <= start:
            break

        local_range = similarities[
            start:end
        ]

        forced_offset = int(
            np.argmin(local_range)
        )

        forced_boundary = (
            start + forced_offset
        )

        if (
            result
            and forced_boundary <= result[-1]
        ):
            break

        result.append(
            forced_boundary
        )

        previous = forced_boundary

    return sorted(
        set(result)
    )


def detect_boundaries(
    embeddings,
    min_segment_windows=12,
    max_segment_windows=100,
    smoothing_radius=2,
    context_radius=4
):
    """
    Detect narrative boundaries using semantic context change.

    Pipeline:

        embeddings
            ↓
        consecutive cosine similarity
            ↓
        similarity smoothing
            ↓
        broader context comparison
            ↓
        adaptive thresholds
            ↓
        candidate boundaries
            ↓
        minimum-distance filtering
            ↓
        maximum-length protection

    Args:
        embeddings (np.ndarray):
            Window embeddings.

        min_segment_windows (int):
            Minimum distance between selected boundaries.

        max_segment_windows (int):
            Maximum allowed segment length.

        smoothing_radius (int):
            Radius for similarity smoothing.

        context_radius (int):
            Number of windows compared on each side.

    Returns:
        tuple:
            boundaries
            raw consecutive similarities
    """

    if len(embeddings) < 2:
        return [], np.array([])

    # --------------------------------------------------------
    # 1. Consecutive semantic similarity
    # --------------------------------------------------------

    similarities = (
        _calculate_consecutive_similarity(
            embeddings
        )
    )

    # --------------------------------------------------------
    # 2. Smooth the similarity curve
    # --------------------------------------------------------

    smoothed = _smooth_similarity(
        similarities,
        smoothing_radius
    )

    # --------------------------------------------------------
    # 3. Calculate broader context change
    # --------------------------------------------------------

    context_change = (
        _calculate_context_change(
            embeddings,
            radius=context_radius
        )
    )

    # --------------------------------------------------------
    # 4. Adaptive thresholds
    # --------------------------------------------------------

    similarity_mean = np.mean(
        smoothed
    )

    similarity_std = np.std(
        smoothed
    )

    similarity_threshold = (
        similarity_mean
        - 0.90 * similarity_std
    )

    valid_context = context_change[
        context_change > 0
    ]

    if len(valid_context) > 0:

        context_mean = np.mean(
            valid_context
        )

        context_std = np.std(
            valid_context
        )

        context_threshold = (
            context_mean
            + 0.50 * context_std
        )

    else:

        context_threshold = 0.0

    # --------------------------------------------------------
    # 5. Candidate boundaries
    # --------------------------------------------------------

    candidates = _find_candidate_boundaries(
        smoothed,
        context_change,
        similarity_threshold,
        context_threshold,
        local_radius=smoothing_radius
    )

    # --------------------------------------------------------
    # 6. Select strongest boundaries
    # --------------------------------------------------------

    boundaries = _select_boundaries(
        candidates,
        smoothed,
        context_change,
        min_segment_windows
    )

    # --------------------------------------------------------
    # 7. Prevent extremely long segments
    # --------------------------------------------------------

    boundaries = _add_max_length_boundaries(
        boundaries,
        smoothed,
        len(embeddings),
        max_segment_windows
    )

    return boundaries, similarities


# ============================================================
# SEGMENT CONSTRUCTION
# ============================================================

def build_segments(windows, boundaries):
    """
    Convert boundary indices into narrative segments.

    Args:
        windows (list[dict]): Semantic transcript windows.
        boundaries (list[int]): Boundary indices.

    Returns:
        list[dict]: Narrative segments.
    """

    if not windows:
        return []

    segments = []

    start_window = 0

    for boundary in sorted(boundaries):

        end_window = min(
            boundary,
            len(windows) - 1
        )

        if end_window < start_window:
            continue

        segment_windows = windows[
            start_window:end_window + 1
        ]

        text = " ".join(
            window["text"]
            for window in segment_windows
        )

        segments.append(
            {
                "text": text,
                "start_pos": start_window,
                "end_pos": end_window,
                "start": segment_windows[0]["start"],
                "end": segment_windows[-1]["end"],
            }
        )

        start_window = (
            end_window + 1
        )

    # Final segment.
    if start_window < len(windows):

        segment_windows = windows[
            start_window:
        ]

        text = " ".join(
            window["text"]
            for window in segment_windows
        )

        segments.append(
            {
                "text": text,
                "start_pos": start_window,
                "end_pos": len(windows) - 1,
                "start": segment_windows[0]["start"],
                "end": segment_windows[-1]["end"],
            }
        )

    return segments


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(segments):
    """
    Extract named entities from each narrative segment.
    """

    nlp = load_nlp_model()

    for segment in segments:

        doc = nlp(
            segment["text"]
        )

        entities = {
            "people": [],
            "places": [],
            "numbers": [],
            "organizations": []
        }

        for ent in doc.ents:

            item = {
                "name": ent.text,
                "label": ent.label_
            }

            if ent.label_ == "PERSON":

                entities["people"].append(
                    item
                )

            elif ent.label_ in {
                "GPE",
                "LOC",
                "FAC"
            }:

                entities["places"].append(
                    item
                )

            elif ent.label_ == "ORG":

                entities["organizations"].append(
                    item
                )

            elif ent.label_ in {
                "CARDINAL",
                "ORDINAL",
                "PERCENT",
                "QUANTITY"
            }:

                entities["numbers"].append(
                    {
                        "value": ent.text,
                        "label": ent.label_
                    }
                )

        segment["entities"] = entities

    return segments


# ============================================================
# TOPIC CLASSIFICATION
# ============================================================

TOPIC_DESCRIPTIONS = {

    "Politics": (
        "politics, government, elections, parliament, "
        "ministers, political parties, legislation and policy"
    ),

    "International News": (
        "international affairs, foreign countries, diplomacy, "
        "international relations, wars and global events"
    ),

    "Sports": (
        "sports, cricket, football, tennis, matches, players, "
        "teams, tournaments and championships"
    ),

    "Weather": (
        "weather forecast, rainfall, temperature, storms, "
        "cyclone, heatwave and weather warnings"
    ),

    "Business": (
        "business, economy, finance, stock markets, companies, "
        "banking, investment and economic news"
    ),

    "Technology": (
        "technology, artificial intelligence, software, computers, "
        "internet, cybersecurity and digital technology"
    ),

    "Health": (
        "healthcare, hospitals, doctors, diseases, medicine, "
        "vaccines and public health"
    ),

    "Entertainment": (
        "movies, films, music, actors, celebrities, television "
        "and entertainment"
    ),

    "Local News": (
        "local city news, transport, traffic, municipal services "
        "and local infrastructure"
    )
}


@lru_cache(maxsize=1)
def create_topic_embeddings():
    """
    Create embeddings for predefined topic descriptions.
    """

    model = load_embedding_model()

    topics = list(
        TOPIC_DESCRIPTIONS.keys()
    )

    descriptions = [
        TOPIC_DESCRIPTIONS[topic]
        for topic in topics
    ]

    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True
    )

    return topics, embeddings


def classify_topic(text):
    """
    Assign the most semantically similar predefined topic.
    """

    model = load_embedding_model()

    topics, topic_embeddings = (
        create_topic_embeddings()
    )

    text_embedding = model.encode(
        [text],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        text_embedding,
        topic_embeddings
    )[0]

    best_index = int(
        np.argmax(scores)
    )

    return topics[best_index]


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(
    text,
    max_keywords=5
):
    """
    Extract simple frequency-based keywords.
    """

    nlp = load_nlp_model()

    doc = nlp(text)

    candidates = []

    for token in doc:

        if not token.is_alpha:
            continue

        if len(token.text) < 4:
            continue

        if token.is_stop:
            continue

        if token.pos_ not in {
            "NOUN",
            "PROPN",
            "ADJ"
        }:
            continue

        candidates.append(
            token.text.lower()
        )

    counts = Counter(
        candidates
    )

    return [
        word
        for word, _ in counts.most_common(
            max_keywords
        )
    ]


# ============================================================
# SEGMENT ENRICHMENT
# ============================================================

def enrich_segments(segments):
    """
    Add topic, subtopic and keywords to every segment.
    """

    for segment in segments:

        segment["topic"] = (
            classify_topic(
                segment["text"]
            )
        )

        segment["keywords"] = (
            extract_keywords(
                segment["text"]
            )
        )

        segment["subtopic"] = (
            " ".join(
                word.title()
                for word in segment[
                    "keywords"
                ][:3]
            )
        )

    return segments


# ============================================================
# CONFIDENCE ESTIMATION
# ============================================================

def calculate_confidence(
    segments,
    boundaries,
    similarities
):
    """
    Estimate segmentation confidence.

    This is a heuristic score, not a calibrated probability.
    """

    if not segments:
        return []

    if len(similarities) == 0:

        for segment in segments:
            segment["confidence"] = 0.5

        return segments

    boundary_scores = []

    for boundary in boundaries:

        if (
            0 <= boundary
            < len(similarities)
        ):

            boundary_scores.append(
                similarities[boundary]
            )

    if boundary_scores:

        min_score = min(
            boundary_scores
        )

        max_score = max(
            boundary_scores
        )

    else:

        min_score = 0.0
        max_score = 1.0

    for index, segment in enumerate(
        segments
    ):

        confidence = 0.5

        if index < len(boundaries):

            boundary = boundaries[
                index
            ]

            if (
                0 <= boundary
                < len(similarities)
            ):

                similarity = (
                    similarities[
                        boundary
                    ]
                )

                if max_score > min_score:

                    normalized = (
                        (
                            similarity
                            - min_score
                        )
                        /
                        (
                            max_score
                            - min_score
                        )
                    )

                    confidence = (
                        1.0
                        - normalized
                    )

        segment["confidence"] = round(
            float(
                np.clip(
                    confidence,
                    0.0,
                    1.0
                )
            ),
            3
        )

    return segments


# ============================================================
# FINAL OUTPUT
# ============================================================

def _format_timestamp(seconds):
    """
    Convert seconds to m:ss format.
    """

    seconds = int(seconds)

    minutes = (
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    return (
        f"{minutes}:"
        f"{remaining_seconds:02d}"
    )


def build_final_output(segments):
    """
    Build the final JSON-compatible project output.
    """

    if not segments:

        return {
            "bulletin_metadata": {
                "source": "YouTube News Bulletin",
                "date": None,
                "duration_minutes": 0,
                "total_stories": 0
            },

            "stories": [],

            "segmentation_quality": {
                "total_segments": 0,
                "average_confidence": 0,
                "topics_identified": [],
                "requires_manual_review": []
            }
        }

    stories = []

    for index, segment in enumerate(
        segments,
        start=1
    ):

        story = {

            "story_id": (
                f"story_{index:03d}"
            ),

            "topic": segment.get(
                "topic",
                "Unknown"
            ),

            "subtopic": segment.get(
                "subtopic",
                ""
            ),

            "timestamp_start": (
                _format_timestamp(
                    segment["start"]
                )
            ),

            "timestamp_end": (
                _format_timestamp(
                    segment["end"]
                )
            ),

            "confidence_score": (
                segment.get(
                    "confidence",
                    0.5
                )
            ),

            "raw_text": segment[
                "text"
            ],

            "entities": segment.get(
                "entities",
                {
                    "people": [],
                    "places": [],
                    "numbers": [],
                    "organizations": []
                }
            ),

            "keywords": segment.get(
                "keywords",
                []
            )
        }

        stories.append(
            story
        )

    confidences = [
        story["confidence_score"]
        for story in stories
    ]

    topics = sorted(
        set(
            story["topic"]
            for story in stories
        )
    )

    requires_manual_review = [
        story["story_id"]
        for story in stories
        if story["confidence_score"] < 0.35
    ]

    duration_minutes = (
        segments[-1]["end"]
        / 60
    )

    return {

        "bulletin_metadata": {

            "source": (
                "YouTube News Bulletin"
            ),

            "date": None,

            "duration_minutes": round(
                duration_minutes,
                2
            ),

            "total_stories": len(
                stories
            )
        },

        "stories": stories,

        "segmentation_quality": {

            "total_segments": len(
                stories
            ),

            "average_confidence": round(
                float(
                    np.mean(
                        confidences
                    )
                ),
                3
            ),

            "topics_identified": topics,

            "requires_manual_review": (
                requires_manual_review
            )
        }
    }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def analyze_video(video_id):
    """
    Run the complete NLP pipeline.

    Pipeline:

        YouTube URL
            ↓
        Transcript extraction
            ↓
        75-word semantic windows
            ↓
        Sentence embeddings
            ↓
        Semantic context comparison
            ↓
        Unsupervised boundary detection
            ↓
        Narrative segments
            ↓
        Confidence estimation
            ↓
        spaCy entity extraction
            ↓
        Semantic topic classification
            ↓
        Keyword extraction
            ↓
        Structured JSON output
    """

    # --------------------------------------------------------
    # 1. Fetch transcript
    # --------------------------------------------------------

    transcript = extract_transcript(
        video_id
    )

    # --------------------------------------------------------
    # 2. Create semantic windows
    # --------------------------------------------------------

    windows = create_windows(
        transcript,
        window_size=75
    )

    # --------------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        windows
    )

    # --------------------------------------------------------
    # 4. Detect narrative boundaries
    # --------------------------------------------------------

    boundaries, similarities = (
        detect_boundaries(
            embeddings,
            min_segment_windows=12,
            max_segment_windows=100
        )
    )

    # --------------------------------------------------------
    # 5. Build narrative segments
    # --------------------------------------------------------

    segments = build_segments(
        windows,
        boundaries
    )

    # --------------------------------------------------------
    # 6. Estimate segmentation confidence
    # --------------------------------------------------------

    segments = calculate_confidence(
        segments,
        boundaries,
        similarities
    )

    # --------------------------------------------------------
    # 7. Extract named entities
    # --------------------------------------------------------

    segments = extract_entities(
        segments
    )

    # --------------------------------------------------------
    # 8. Add topics and keywords
    # --------------------------------------------------------

    segments = enrich_segments(
        segments
    )

    # --------------------------------------------------------
    # 9. Add internal metadata
    # --------------------------------------------------------

    for index, segment in enumerate(
        segments,
        start=1
    ):

        segment["segment_id"] = index

        segment["word_count"] = len(
            segment["text"].split()
        )

    # --------------------------------------------------------
    # 10. Build final JSON result
    # --------------------------------------------------------

    return build_final_output(
        segments
    )