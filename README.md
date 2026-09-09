````markdown
# Unsupervised Narrative Segmentation & Entity Extraction from Noisy ASR Transcripts

An end-to-end Natural Language Processing (NLP) pipeline for transforming long, noisy, automatically generated YouTube news transcripts into structured narrative stories with timestamps, topics, keywords, and named entities.

---

## 📌 Project Overview

Long-form news videos often contain multiple news stories presented continuously without explicit boundaries.

Automatically generated YouTube transcripts make this problem more difficult because they commonly contain:

- Missing or unreliable punctuation
- Incorrect words caused by Automatic Speech Recognition (ASR)
- Poor sentence boundaries
- Repeated phrases
- Long continuous streams of text
- No explicit indication of where one story ends and another begins

This project addresses this problem using an **unsupervised semantic narrative segmentation approach**.

Instead of requiring manually labelled story boundaries, the system:

1. Extracts the transcript from a YouTube video.
2. Preprocesses and explores the transcript.
3. Divides the transcript into fixed-size semantic windows.
4. Converts each window into a sentence embedding.
5. Calculates semantic similarity between consecutive windows.
6. Detects potential narrative boundaries from semantic changes.
7. Constructs timestamped narrative segments.
8. Assigns semantic topic categories to the segments.
9. Extracts named entities using Named Entity Recognition (NER).
10. Extracts important keywords.
11. Estimates a confidence score for each segment.
12. Produces structured JSON output.
13. Displays the results through a Streamlit web application.

---

## 🎯 Problem Statement

Given a long YouTube news video containing noisy ASR-generated subtitles:

> **Automatically identify meaningful narrative/story boundaries and extract useful structured information from each story without relying on manually labelled segmentation data.**

The system is designed for long-form news content where multiple topics can occur within the same transcript.

---

## 💡 Core Idea

The central assumption of this project is:

> **When the narrative changes, the semantic similarity between neighbouring portions of the transcript is likely to decrease.**

For example:

```text
Window A:
"The government introduced a new climate bill..."

                    ↓

              HIGH SIMILARITY

                    ↓

Window B:
"Parliament members debated the proposed legislation..."
````

If the next portion discusses a substantially different subject:

```text
Window A:
"The government introduced a new climate bill..."

                    ↓

              LOWER SIMILARITY

                    ↓

Window B:
"The cricket team announced its squad for the upcoming match..."
```

the decrease in semantic similarity can act as a signal for a possible narrative boundary.

The project therefore treats narrative segmentation as a **semantic change detection problem**.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │     YouTube URL      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Transcript Extraction│
                         │ youtube-transcript-  │
                         │ api                  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Preprocessing & EDA  │
                         │                      │
                         │ Words / Characters  │
                         │ Snippets / Timing   │
                         │ Punctuation         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Semantic Windowing   │
                         │                      │
                         │ Fixed 75-word       │
                         │ windows              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Sentence Embeddings  │
                         │ all-MiniLM-L6-v2     │
                         │ 384 dimensions       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Consecutive Cosine   │
                         │ Similarity           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Boundary Detection   │
                         │                      │
                         │ Statistical          │
                         │ Thresholding         │
                         │ Local Minima         │
                         │ Context Comparison   │
                         │ Segment Constraints  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Narrative Segments   │
                         │ + Timestamps         │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
          ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
          │ Topic          │ │ Named Entity │ │ Keyword      │
          │ Classification │ │ Recognition  │ │ Extraction   │
          └───────┬────────┘ └──────┬───────┘ └──────┬───────┘
                  │                 │                │
                  └─────────────────┼────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Confidence           │
                         │ Estimation            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Structured JSON      │
                         │ Output               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Application│
                         └──────────────────────┘
```

---

# ⚙️ Processing Pipeline

## 1. YouTube Transcript Extraction

The system accepts a YouTube video URL and extracts its automatically generated transcript using `youtube-transcript-api`.

Each transcript snippet contains:

* Text
* Start timestamp
* Duration

These timestamps are preserved so that the final narrative segments can be mapped back to the original video.

---

## 2. Transcript Exploration and Preprocessing

Before segmentation, the transcript is explored to understand the characteristics of the ASR data.

The analysis includes:

* Number of transcript snippets
* Total characters
* Total words
* Average snippet length
* Minimum and maximum snippet length
* Punctuation frequency
* Approximate sentence statistics
* Timestamp distribution
* Overlapping transcript snippets

This exploratory stage is important because ASR transcripts do not behave like clean written documents.

For the evaluated transcript:

* Approximately **14,409 transcript snippets**
* Approximately **92,133 words**
* Approximately **512,923 characters**

The analysis showed that punctuation-based sentence splitting was unreliable, motivating the use of fixed-size semantic windows.

---

## 3. Semantic Windowing

Instead of relying on ASR punctuation or sentence boundaries, the transcript is divided into fixed-size windows.

The current implementation uses:

```text
75 words per window
```

Conceptually:

```text
Transcript
    │
    ├── Window 1 → 75 words
    ├── Window 2 → 75 words
    ├── Window 3 → 75 words
    ├── Window 4 → 75 words
    └── ...
```

Each window retains:

* Window ID
* Text
* Start timestamp
* End timestamp

For the evaluated transcript, this produced approximately **1,183 semantic windows**.

### Why fixed-size windows?

ASR punctuation and sentence boundaries can be unreliable.

Fixed-size windows provide a consistent unit for semantic comparison and prevent the segmentation process from depending entirely on noisy punctuation.

---

## 4. Sentence Embeddings

Each semantic window is converted into a numerical vector using:

**`all-MiniLM-L6-v2`**

from the Sentence Transformers library.

The model represents each text window as a **384-dimensional embedding**.

Conceptually:

```text
Text Window
     │
     ▼
Sentence Transformer
     │
     ▼
384-dimensional vector
```

For example:

```text
"Parliament debated the climate bill..."
                    │
                    ▼
[0.12, -0.04, 0.31, ..., 0.08]
```

The embedding represents the semantic information of the text in vector space.

---

## 5. Consecutive Cosine Similarity

Once embeddings are generated, the system compares neighbouring windows.

```text
Window 1 ↔ Window 2
Window 2 ↔ Window 3
Window 3 ↔ Window 4
...
```

Cosine similarity measures the similarity between two embedding vectors based on their orientation.

The formula is:

```text
                 A · B
similarity = ─────────────
             ||A|| ||B||
```

Higher similarity generally indicates that neighbouring windows discuss semantically related content.

A possible narrative transition may therefore look like:

```text
High similarity
High similarity
High similarity
High similarity
      ↓
LOW similarity  ← possible boundary
      ↓
High similarity
High similarity
```

---

## 6. Unsupervised Boundary Detection

The project does not require manually labelled story boundaries.

Potential boundaries are derived statistically from the similarity distribution.

The basic threshold is:

```text
τ = μ - ασ
```

where:

* `τ` = similarity threshold
* `μ` = mean similarity
* `σ` = standard deviation
* `α` = sensitivity parameter

Similarity values significantly below the normal similarity level become candidate boundaries.

However, an individual low similarity value does not automatically create a new story.

The implementation therefore considers several additional constraints.

### Local Minima

A candidate boundary should represent a local drop in similarity rather than simply an isolated low value.

### Smoothed Similarity

Neighbouring similarity values are considered together to reduce sensitivity to isolated noisy ASR effects.

### Context Comparison

Broader neighbouring context is considered when evaluating potential semantic transitions.

### Minimum Segment Size

Closely spaced boundaries are filtered to prevent extremely small and meaningless segments.

### Maximum Segment Size

A maximum window gap prevents the system from producing one excessively large segment when the similarity signal does not provide a boundary.

---

## 7. Narrative Segment Construction

After candidate boundaries are selected, the semantic windows are combined into larger narrative segments.

Each segment contains:

* Segment ID
* Start timestamp
* End timestamp
* Text
* Word count

Conceptually:

```text
Window 1
Window 2
Window 3
Window 4
     │
     └──── Segment 1

Window 5
Window 6
Window 7
     │
     └──── Segment 2
```

The original transcript timestamps are preserved so that each segment can be associated with a portion of the source video.

---

## 8. Semantic Topic Classification

Each narrative segment is assigned a broad semantic topic.

The current topic categories include:

* Politics
* International News
* Sports
* Weather
* Business
* Technology
* Health
* Entertainment
* Local News

Instead of using simple keyword rules, each topic is represented by a semantic description.

For example:

```text
Sports:
"sports, cricket, football, tennis, matches,
players, teams, tournaments and championships"
```

The topic descriptions are converted into embeddings using the same embedding model.

The segment embedding is then compared against the topic embeddings.

The topic with the highest semantic similarity is selected.

This is more flexible than a rule such as:

```text
if "cricket" in text:
    topic = "Sports"
```

---

## 9. Named Entity Recognition

Named Entity Recognition (NER) is performed using:

**spaCy `en_core_web_sm`**

The system extracts entities such as:

* People
* Places
* Organizations
* Numbers

Example:

```text
PERSON
    Rahul Gandhi
    Virat Kohli

GPE / LOC
    India
    Delhi

ORG
    Parliament
    BJP
```

Numerical information such as:

```text
87 runs
42 degrees
15 districts
```

is also retained for structured extraction.

Because the input is noisy ASR text, entity recognition can occasionally produce incorrect classifications. This is an expected limitation when applying a general-purpose pretrained NER model to speech transcripts.

---

## 10. Keyword Extraction

Important keywords are extracted from each narrative segment.

The current approach uses spaCy tokenization and considers words that are:

* Alphabetic
* At least four characters long
* Not stop words
* Nouns, proper nouns, or adjectives

Candidate keywords are ranked using their frequency within the segment.

The top keywords are retained for each story and are also used to help generate a short descriptive subtopic label.

---

## 11. Confidence Estimation

Each narrative segment receives a heuristic confidence score.

The confidence is derived from the semantic similarity around detected boundaries.

The intuition is:

```text
Stronger semantic drop
        ↓
More likely narrative transition
        ↓
Higher segmentation confidence
```

This score should be interpreted as a **relative confidence indicator**, not as a calibrated probability.

A production system would ideally calibrate this score using a manually labelled evaluation dataset.

---

# 📄 Structured Output

The pipeline produces structured JSON containing:

* Bulletin metadata
* Story IDs
* Topics
* Subtopics
* Timestamps
* Confidence scores
* Raw transcript text
* Named entities
* Keywords
* Segmentation quality information

Example:

```json
{
  "bulletin_metadata": {
    "source": "YouTube News Bulletin",
    "date": null,
    "duration_minutes": 30.0,
    "total_stories": 6
  },
  "stories": [
    {
      "story_id": "story_001",
      "topic": "Politics",
      "subtopic": "Climate Bill Vote",
      "timestamp_start": "0:05",
      "timestamp_end": "1:15",
      "confidence_score": 0.89,
      "raw_text": "...",
      "entities": {
        "people": [],
        "places": [],
        "numbers": [],
        "organizations": []
      },
      "keywords": [
        "climate",
        "bill",
        "parliament"
      ]
    }
  ],
  "segmentation_quality": {
    "total_segments": 6,
    "average_confidence": 0.88,
    "topics_identified": [
      "Politics",
      "Sports",
      "Weather"
    ],
    "requires_manual_review": []
  }
}
```

The exact number of segments depends on the input transcript and detected semantic boundaries.

---

# 📈 Exploratory Data Analysis

The project includes:

```text
notebooks/01_data_exploration.ipynb
```

The notebook is used to inspect:

* Transcript size
* Word distribution
* Character distribution
* Snippet lengths
* Punctuation
* Timestamp information
* Semantic windowing
* Embedding generation
* Similarity distributions
* Candidate boundaries
* Segment statistics
* Extracted entities

This provides visibility into the intermediate stages of the pipeline rather than treating the segmentation process as a black box.

---

# 🖥️ Streamlit Application

A Streamlit interface provides a frontend for the complete pipeline.

The application accepts a YouTube video URL and executes:

```text
YouTube Video URL
        │
        ▼
   Analyze Video
        │
        ▼
   NLP Pipeline
        │
        ▼
Narrative Segments
```

The interface displays:

* Number of detected stories
* Video duration
* Number of detected topics
* Individual story segments
* Start and end timestamps
* Topic
* Subtopic
* Transcript
* Extracted entities
* Keywords

---

# 📁 Project Structure

```text
Unsupervised-Narrative-Segmentation-Entity-Extraction-Project/
│
├── app.py
│
├── src/
│   └── pipeline.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── data/
│       ├── entities.csv
│       ├── final_segments.json
│       └── narrative_segments.csv
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Asmitha-M-2006/Unsupervised-Narrative-Segmentation-Entity-Extraction-Project.git
```

Move into the project directory:

```bash
cd Unsupervised-Narrative-Segmentation-Entity-Extraction-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

Enter a YouTube video URL and click:

```text
Analyze Video
```

The complete NLP pipeline will run and the resulting narrative segments will be displayed.

---

# 🔬 End-to-End Workflow

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Transcript Exploration
      │
      ▼
75-word Semantic Windows
      │
      ▼
384-dimensional Embeddings
      │
      ▼
Cosine Similarity
      │
      ▼
Unsupervised Boundary Detection
      │
      ▼
Narrative Segments
      │
      ├───────────────┐
      ▼               ▼
Topic Classification  NER
      │               │
      └───────┬───────┘
              ▼
      Keyword Extraction
              │
              ▼
      Confidence Estimation
              │
              ▼
        Structured JSON
              │
              ▼
        Streamlit UI
```

---

# ⚖️ Design Decisions

## Why Fixed-Size Windows?

ASR transcripts often contain unreliable punctuation and sentence boundaries.

Fixed-size windows provide a consistent unit for semantic comparison.

### Alternative

Sentence-based segmentation could be used, but it would depend heavily on reliable punctuation or sentence detection.

---

## Why Sentence Embeddings?

Traditional keyword-based approaches can fail when two pieces of text discuss the same subject using different vocabulary.

Semantic embeddings capture broader contextual meaning.

### Alternative

TF-IDF or Bag-of-Words could be used, but these approaches primarily represent lexical overlap rather than semantic similarity.

---

## Why Cosine Similarity?

Cosine similarity is well suited for comparing text embeddings because it measures the orientation of vectors rather than their absolute magnitude.

It is also computationally simple and widely used for semantic similarity.

---

## Why Unsupervised Segmentation?

Manually labelled story boundaries are expensive to create.

An unsupervised approach allows the system to discover potential narrative transitions directly from the transcript.

### Trade-off

The approach cannot guarantee that every detected boundary corresponds to a true editorial story boundary.

---

# ⚠️ Limitations

## 1. ASR Noise

Incorrect transcription can affect:

* Embeddings
* Topic classification
* NER
* Keyword extraction

## 2. Semantic Similarity Is Not Perfect

A drop in similarity does not always mean a new story.

A single story may contain several semantically different sections, while different stories may sometimes discuss related subjects.

## 3. Hyperparameter Sensitivity

Parameters such as:

* Window size
* Similarity threshold
* Smoothing radius
* Minimum segment size
* Maximum segment size

can influence segmentation results.

## 4. NER Errors

The pretrained spaCy model was not specifically trained for noisy ASR transcripts.

Some entities may therefore be incorrectly classified.

## 5. Heuristic Confidence

The confidence score is not currently calibrated against ground-truth segmentation labels.

## 6. No Ground-Truth Evaluation Dataset

A manually annotated dataset containing true story boundaries would allow quantitative evaluation using metrics such as:

* Precision
* Recall
* F1-score
* Boundary detection accuracy
* WindowDiff
* Pk

---

# 🔮 Future Improvements

Potential improvements include:

* Overlapping semantic windows
* Adaptive window sizes
* Improved boundary scoring
* Change-point detection algorithms
* Supervised or weakly supervised segmentation
* ASR error correction
* Stronger transformer embedding models
* Entity normalization
* Entity coreference resolution
* Entity role and relationship extraction
* Improved subtopic generation
* Manual boundary correction through the UI
* Evaluation against manually labelled stories
* Similarity and boundary visualizations
* JSON and CSV export directly from Streamlit
* Public deployment

---

# 🧰 Technologies & Skills

## Programming

* Python

## Natural Language Processing

* Automatic Speech Recognition (ASR) transcript processing
* Text preprocessing
* Semantic windowing
* Sentence embeddings
* Cosine similarity
* Named Entity Recognition
* Keyword extraction
* Semantic topic classification
* Narrative segmentation

## Machine Learning

* Unsupervised segmentation
* Statistical thresholding
* Similarity-based change detection
* Semantic classification

## Python Libraries

* NumPy
* pandas
* scikit-learn
* spaCy
* Sentence Transformers
* youtube-transcript-api
* Streamlit

## Data Processing

* Transcript analysis
* Timestamp processing
* Exploratory Data Analysis
* Structured JSON generation
* CSV generation

## Software Engineering

* Modular Python pipeline
* Virtual environments
* Dependency management
* Git
* GitHub

---

# 👨‍🏫 Project Guidance

**Assigned Mentor:** Parth Dhola

**Mentor:** Amaan Irfan Sir

**Faculty Advisor:** Prof. Prithwijit Guha

---

# 📌 Project Status

The core NLP pipeline and Streamlit interface have been implemented.

Current work focuses on:

* Final validation
* Documentation
* GitHub presentation
* Project demonstration
* Architecture documentation
* Final report and submission preparation

---

# 🎓 Academic Context

This project explores how unsupervised semantic methods can be applied to narrative segmentation of noisy, automatically generated news transcripts.

The project combines:

```text
NLP
+
Sentence Embeddings
+
Unsupervised Learning
+
Semantic Similarity
+
Information Extraction
+
Data Analysis
+
Web Application Development
```

to transform unstructured news transcripts into structured narrative information.

---

# 🔗 Project Links

## GitHub Repository

[https://github.com/Asmitha-M-2006/Unsupervised-Narrative-Segmentation-Entity-Extraction-Project](https://github.com/Asmitha-M-2006/Unsupervised-Narrative-Segmentation-Entity-Extraction-Project)

## Project Presentation Video

*To be added after the final screen-recorded presentation is uploaded.*

## GitHub Project Website

*To be added after GitHub Pages is configured.*

---

# 📜 License

This project is developed for academic and educational purposes.

```
```
