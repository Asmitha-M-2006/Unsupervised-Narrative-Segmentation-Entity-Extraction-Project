from collections import Counter
from functools import lru_cache
import re

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi


# --------------------------------------------------
# 1. Model Loading
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_nlp_model():
    return spacy.load("en_core_web_sm")


# --------------------------------------------------
# 2. Transcript Extraction
# --------------------------------------------------

def extract_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    return transcript


# --------------------------------------------------
# 3. Fixed-size Semantic Windows
# --------------------------------------------------

def create_windows(transcript, window_size=75):
    windows = []

    current_words = []
    window_start = None
    window_end = None

    for snippet in transcript:
        words = snippet.text.split()

        if window_start is None:
            window_start = snippet.start

        current_words.extend(words)
        window_end = snippet.start + snippet.duration

        if len(current_words) >= window_size:
            windows.append({
                "id": len(windows),
                "text": " ".join(current_words),
                "start": window_start,
                "end": window_end
            })

            current_words = []
            window_start = None
            window_end = None

    if current_words:
        windows.append({
            "id": len(windows),
            "text": " ".join(current_words),
            "start": window_start,
            "end": window_end
        })

    return windows


# --------------------------------------------------
# 4. Sentence Embeddings
# --------------------------------------------------

def create_embeddings(windows):
    model = load_embedding_model()

    texts = [window["text"] for window in windows]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


# --------------------------------------------------
# 5. Unsupervised Boundary Detection
# --------------------------------------------------

def detect_boundaries(
    embeddings,
    min_segment_windows=8
):
    similarities = []

    for i in range(len(embeddings) - 1):

        score = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]

        similarities.append(score)

    similarities = np.array(similarities)

    if len(similarities) == 0:
        return np.array([], dtype=int), similarities

    threshold = (
        np.mean(similarities)
        - np.std(similarities)
    )

    candidates = np.where(
        similarities < threshold
    )[0]

    filtered_boundaries = []

    for boundary in candidates:

        if (
            not filtered_boundaries
            or boundary
            - filtered_boundaries[-1]
            >= min_segment_windows
        ):
            filtered_boundaries.append(boundary)

    return (
        np.array(filtered_boundaries),
        similarities
    )


# --------------------------------------------------
# 6. Build Narrative Segments
# --------------------------------------------------

def build_segments(windows, boundaries):
    segments = []

    start_window = 0

    for boundary in boundaries:

        end_window = boundary

        segments.append({
            "start": windows[start_window]["start"],
            "end": windows[end_window]["end"],
            "text": " ".join(
                windows[i]["text"]
                for i in range(
                    start_window,
                    end_window + 1
                )
            )
        })

        start_window = boundary + 1

    if start_window < len(windows):

        segments.append({
            "start": windows[start_window]["start"],
            "end": windows[-1]["end"],
            "text": " ".join(
                windows[i]["text"]
                for i in range(
                    start_window,
                    len(windows)
                )
            )
        })

    return segments


# --------------------------------------------------
# 7. Named Entity Recognition
# --------------------------------------------------

def extract_entities(segments):
    nlp = load_nlp_model()

    for segment in segments:

        doc = nlp(segment["text"])

        entities = {
            "people": [],
            "places": [],
            "numbers": [],
            "organizations": []
        }

        for ent in doc.ents:

            if ent.label_ == "PERSON":

                entities["people"].append({
                    "name": ent.text,
                    "mentions": 1
                })

            elif ent.label_ in [
                "GPE",
                "LOC",
                "FAC"
            ]:

                entities["places"].append({
                    "name": ent.text,
                    "type": ent.label_,
                    "mentions": 1
                })

            elif ent.label_ == "ORG":

                entities["organizations"].append({
                    "name": ent.text,
                    "type": "organization"
                })

            elif ent.label_ in [
                "CARDINAL",
                "ORDINAL",
                "PERCENT",
                "QUANTITY"
            ]:

                entities["numbers"].append({
                    "value": ent.text,
                    "type": ent.label_
                })

        segment["entities"] = entities

    return segments


# --------------------------------------------------
# 8. Topic Classification
# --------------------------------------------------

TOPIC_KEYWORDS = {
    "Politics": [
        "government",
        "minister",
        "parliament",
        "election",
        "political",
        "opposition",
        "vote",
        "policy",
        "party",
        "prime minister",
        "congress"
    ],

    "International News": [
        "international",
        "foreign",
        "united states",
        "america",
        "china",
        "india",
        "pakistan",
        "russia",
        "war",
        "tariff",
        "trade",
        "president"
    ],

    "Sports": [
        "cricket",
        "football",
        "tennis",
        "match",
        "player",
        "team",
        "runs",
        "wicket",
        "goal",
        "championship",
        "tournament",
        "win",
        "lost"
    ],

    "Weather": [
        "weather",
        "temperature",
        "rain",
        "rainfall",
        "forecast",
        "storm",
        "cyclone",
        "heat",
        "degrees",
        "celsius",
        "wind"
    ],

    "Business": [
        "business",
        "market",
        "stock",
        "company",
        "economy",
        "economic",
        "profit",
        "bank",
        "investment",
        "finance"
    ],

    "Technology": [
        "technology",
        "ai",
        "artificial intelligence",
        "software",
        "computer",
        "digital",
        "internet",
        "robot",
        "startup"
    ],

    "Health": [
        "health",
        "hospital",
        "doctor",
        "disease",
        "medicine",
        "medical",
        "patients",
        "virus"
    ],

    "Entertainment": [
        "movie",
        "film",
        "actor",
        "actress",
        "music",
        "celebrity",
        "concert",
        "entertainment"
    ]
}


def classify_topic(text):
    text_lower = text.lower()

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():

        score = 0

        for keyword in keywords:
            score += text_lower.count(keyword)

        scores[topic] = score

    best_topic = max(
        scores,
        key=scores.get
    )

    if scores[best_topic] == 0:
        return "Other"

    return best_topic


# --------------------------------------------------
# 9. Keyword Extraction
# --------------------------------------------------

def extract_keywords(
    text,
    max_keywords=5
):
    words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        text.lower()
    )

    stop_words = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "were",
        "they",
        "their",
        "there",
        "about",
        "will",
        "said",
        "been",
        "would",
        "could",
        "which",
        "what",
        "when",
        "where",
        "also",
        "more",
        "than",
        "into",
        "after",
        "before",
        "news",
        "today"
    }

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    counts = Counter(
        filtered_words
    )

    return [
        word
        for word, count
        in counts.most_common(max_keywords)
    ]


def enrich_segments(segments):

    for segment in segments:

        segment["topic"] = classify_topic(
            segment["text"]
        )

        segment["keywords"] = extract_keywords(
            segment["text"]
        )

    return segments


# --------------------------------------------------
# 10. Confidence Scoring
# --------------------------------------------------

def calculate_confidence(
    segments,
    boundaries,
    similarities
):
    if len(segments) == 0:
        return segments

    boundary_scores = []

    for boundary in boundaries:

        if boundary < len(similarities):
            boundary_scores.append(
                similarities[boundary]
            )

    if boundary_scores:

        min_score = min(boundary_scores)
        max_score = max(boundary_scores)

    else:

        min_score = 0
        max_score = 1

    for i, segment in enumerate(segments):

        scores = []

        # Boundary before this segment
        if i > 0:

            boundary_index = boundaries[i - 1]

            if boundary_index < len(similarities):
                scores.append(
                    similarities[boundary_index]
                )

        # Boundary after this segment
        if i < len(boundaries):

            boundary_index = boundaries[i]

            if boundary_index < len(similarities):
                scores.append(
                    similarities[boundary_index]
                )

        if scores:

            average_boundary_score = np.mean(
                scores
            )

            if max_score > min_score:

                confidence = 1 - (
                    (
                        average_boundary_score
                        - min_score
                    )
                    / (max_score - min_score)
                )

            else:

                confidence = 0.5

        else:

            confidence = 0.5

        segment["confidence_score"] = round(
            float(
                np.clip(
                    confidence,
                    0,
                    1
                )
            ),
            2
        )

    return segments


# --------------------------------------------------
# 11. Build Final JSON Output
# --------------------------------------------------

def build_final_output(segments):

    stories = []

    for i, segment in enumerate(segments):

        keywords = segment.get(
            "keywords",
            []
        )

        stories.append({

            "story_id": (
                f"story_{i + 1:03d}"
            ),

            "topic": segment.get(
                "topic",
                "Other"
            ),

            "subtopic": (
                " ".join(
                    keywords[:3]
                ).title()
                if keywords
                else "News Update"
            ),

            "timestamp_start": (
                segment["start"]
            ),

            "timestamp_end": (
                segment["end"]
            ),

            "confidence_score": segment.get(
                "confidence_score",
                0.0
            ),

            "raw_text": segment["text"],

            "entities": segment.get(
                "entities",
                {}
            ),

            "keywords": keywords
        })

    if segments:

        duration_minutes = (
            segments[-1]["end"] / 60
        )

    else:

        duration_minutes = 0

    topics = list(
        dict.fromkeys(
            story["topic"]
            for story in stories
        )
    )

    confidence_scores = [
        story["confidence_score"]
        for story in stories
    ]

    average_confidence = (
        np.mean(confidence_scores)
        if confidence_scores
        else 0.0
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
                    average_confidence
                ),
                2
            ),

            "topics_identified": topics,

            "requires_manual_review": []
        }
    }


# --------------------------------------------------
# 12. Complete Analysis Pipeline
# --------------------------------------------------

def analyze_video(video_id):

    transcript = extract_transcript(
        video_id
    )

    windows = create_windows(
        transcript
    )

    embeddings = create_embeddings(
        windows
    )

    boundaries, similarities = (
        detect_boundaries(
            embeddings
        )
    )

    segments = build_segments(
        windows,
        boundaries
    )

    segments = calculate_confidence(
        segments,
        boundaries,
        similarities
    )

    segments = extract_entities(
        segments
    )

    segments = enrich_segments(
        segments
    )

    for i, segment in enumerate(
        segments
    ):

        segment["segment_id"] = i

        segment["word_count"] = len(
            segment["text"].split()
        )

    return build_final_output(
        segments
    )