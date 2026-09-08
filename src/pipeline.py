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

    Why:
        ASR snippets are too short and noisy to compare directly.
        Grouping approximately 75 words gives the embedding model
        enough context to represent the underlying topic.

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

    # Preserve the final partial window.
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

    texts = [window["text"] for window in windows]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


# ============================================================
# NARRATIVE SEGMENTATION
# ============================================================

def _find_candidate_boundaries(
    similarities,
    threshold,
    local_radius=2
):
    """
    Find strong local semantic drops.

    A point is considered a candidate boundary when:
        1. Its similarity is below the global threshold.
        2. It is a local minimum compared with nearby windows.

    This is better than simply selecting every similarity value
    below the threshold.
    """

    candidates = []

    for i in range(local_radius, len(similarities) - local_radius):

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

        if current <= np.min(neighborhood):

            candidates.append(i)

    return candidates


def _select_boundaries(
    candidates,
    similarities,
    min_segment_windows
):
    """
    Select candidate boundaries while enforcing a minimum
    distance between consecutive boundaries.

    If multiple candidate boundaries are too close together,
    keep the strongest semantic drop.
    """

    if not candidates:
        return []

    selected = []

    # Process candidates from strongest semantic drop
    # to weakest.
    ranked = sorted(
        candidates,
        key=lambda i: similarities[i]
    )

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

        while boundary - previous > max_segment_windows:

            start = previous + 1

            end = min(
                start + max_segment_windows,
                boundary
            )

            if end <= start:
                break

            local_range = similarities[start:end]

            forced_offset = int(
                np.argmin(local_range)
            )

            forced_boundary = start + forced_offset

            if (
                result
                and forced_boundary <= result[-1]
            ):
                break

            result.append(forced_boundary)
            previous = forced_boundary

        result.append(boundary)
        previous = boundary

    # Handle the final segment.
    while total_windows - 1 - previous > max_segment_windows:

        start = previous + 1

        end = min(
            start + max_segment_windows,
            total_windows - 1
        )

        if end <= start:
            break

        local_range = similarities[start:end]

        forced_offset = int(
            np.argmin(local_range)
        )

        forced_boundary = start + forced_offset

        if (
            result
            and forced_boundary <= result[-1]
        ):
            break

        result.append(forced_boundary)
        previous = forced_boundary

    return sorted(set(result))


def detect_boundaries(
    embeddings,
    min_segment_windows=20,
    max_segment_windows=100,
    smoothing_radius=2
):
    """
    Detect narrative boundaries using consecutive semantic
    similarity.

    Pipeline:

        embeddings
            ↓
        cosine similarity
            ↓
        local smoothing
            ↓
        adaptive threshold
            ↓
        local semantic minima
            ↓
        minimum-distance filtering
            ↓
        maximum-length protection

    Args:
        embeddings (np.ndarray): Window embeddings.
        min_segment_windows (int): Minimum distance between
            narrative boundaries.
        max_segment_windows (int): Maximum allowed segment length.
        smoothing_radius (int): Radius used for local smoothing.

    Returns:
        tuple:
            boundaries: List of boundary indices.
            similarities: Raw consecutive similarities.
    """

    if len(embeddings) < 2:
        return [], np.array([])

    similarities = cosine_similarity(
        embeddings[:-1],
        embeddings[1:]
    ).diagonal()

    similarities = np.asarray(
        similarities,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Smooth the similarity curve.
    #
    # This reduces the effect of one noisy ASR window causing
    # an artificial boundary.
    # --------------------------------------------------------

    kernel_size = (
        smoothing_radius * 2
    ) + 1

    kernel = np.ones(
        kernel_size,
        dtype=np.float32
    ) / kernel_size

    padded = np.pad(
        similarities,
        (
            smoothing_radius,
            smoothing_radius
        ),
        mode="edge"
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    # --------------------------------------------------------
    # Adaptive threshold.
    #
    # We do not use a fixed similarity value because different
    # videos naturally have different similarity distributions.
    # --------------------------------------------------------

    mean_similarity = np.mean(smoothed)
    std_similarity = np.std(smoothed)

    threshold = (
        mean_similarity
        - 0.90 * std_similarity
    )

    # --------------------------------------------------------
    # Candidate boundaries.
    # --------------------------------------------------------

    candidates = _find_candidate_boundaries(
        smoothed,
        threshold,
        local_radius=smoothing_radius
    )

    # --------------------------------------------------------
    # Remove candidates that are too close together.
    # --------------------------------------------------------

    boundaries = _select_boundaries(
        candidates,
        smoothed,
        min_segment_windows
    )

    # --------------------------------------------------------
    # Prevent extremely long stories.
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
        list[dict]: Narrative segments with timestamps.
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

        start_window = end_window + 1

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

    spaCy is used for:
        PERSON
        GPE
        LOC
        FAC
        ORG
        CARDINAL
        ORDINAL
        PERCENT
        QUANTITY

    Args:
        segments (list[dict]): Narrative segments.

    Returns:
        list[dict]: Segments with entity information.
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

                entities["people"].append(item)

            elif ent.label_ in {
                "GPE",
                "LOC",
                "FAC"
            }:

                entities["places"].append(item)

            elif ent.label_ == "ORG":

                entities["organizations"].append(item)

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
    Create embeddings for the predefined topic descriptions.

    Returns:
        tuple:
            topic names
            topic embeddings
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

    Args:
        text (str): Narrative segment text.

    Returns:
        str: Predicted topic.
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

    Uses nouns, proper nouns and adjectives while removing
    stopwords and very short words.

    Args:
        text (str): Segment text.
        max_keywords (int): Maximum keywords to return.

    Returns:
        list[str]: Keywords.
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

    counts = Counter(candidates)

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

        segment["topic"] = classify_topic(
            segment["text"]
        )

        segment["keywords"] = (
            extract_keywords(
                segment["text"]
            )
        )

        segment["subtopic"] = (
            " ".join(
                word.title()
                for word in segment["keywords"][:3]
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
    Estimate confidence for each segment.

    Important:
        This is a heuristic confidence score, NOT a calibrated
        probability.

    Lower similarity at a boundary means a stronger semantic
    transition and therefore higher segmentation confidence.
    """

    if not segments:
        return []

    if len(similarities) == 0:
        for segment in segments:
            segment["confidence"] = 0.5

        return segments

    boundary_scores = []

    for boundary in boundaries:

        if 0 <= boundary < len(similarities):

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

            boundary = boundaries[index]

            if 0 <= boundary < len(similarities):

                similarity = (
                    similarities[boundary]
                )

                if max_score > min_score:

                    normalized = (
                        (similarity - min_score)
                        / (max_score - min_score)
                    )

                    confidence = (
                        1.0 - normalized
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
    Convert seconds into a human-readable timestamp.

    Example:
        65.4 -> 1:05
    """

    seconds = int(seconds)

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return (
        f"{minutes}:"
        f"{remaining_seconds:02d}"
    )


def build_final_output(segments):
    """
    Build the final JSON-compatible project output.

    Returns:
        dict: Final structured result.
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

            "timestamp_start": _format_timestamp(
                segment["start"]
            ),

            "timestamp_end": _format_timestamp(
                segment["end"]
            ),

            "confidence_score": segment.get(
                "confidence",
                0.5
            ),

            "raw_text": segment["text"],

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

        stories.append(story)

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

    # Stories with relatively weak confidence are flagged
    # for manual review.
    requires_manual_review = [
        story["story_id"]
        for story in stories
        if story["confidence_score"] < 0.35
    ]

    duration_minutes = (
        segments[-1]["end"] / 60
    )

    return {
        "bulletin_metadata": {
            "source": "YouTube News Bulletin",
            "date": None,
            "duration_minutes": round(
                duration_minutes,
                2
            ),
            "total_stories": len(stories)
        },

        "stories": stories,

        "segmentation_quality": {
            "total_segments": len(stories),

            "average_confidence": round(
                float(
                    np.mean(confidences)
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
        Semantic similarity
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

    Args:
        video_id (str): YouTube video ID.

    Returns:
        dict: Final structured analysis.
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
            min_segment_windows=20,
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
    # 10. Build final JSON-compatible result
    # --------------------------------------------------------

    return build_final_output(
        segments
    )