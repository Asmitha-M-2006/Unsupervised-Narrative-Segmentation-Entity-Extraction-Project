# Unsupervised Narrative Segmentation & Entity Extraction

An NLP pipeline for converting long, noisy, auto-generated YouTube news transcripts into meaningful narrative segments with timestamps and extracted entities.

## Problem

News reporters often discuss multiple stories continuously without explicit topic boundaries. Automatically generated YouTube transcripts are also noisy and usually lack reliable punctuation and sentence boundaries.

This project uses an unsupervised semantic approach to identify likely narrative boundaries.

## Pipeline

YouTube Transcript
→ Preprocessing & Exploration
→ Fixed-size Word Windows
→ Sentence Embeddings
→ Consecutive Cosine Similarity
→ Unsupervised Boundary Detection
→ Narrative Segmentation
→ Named Entity Recognition
→ Structured JSON Output

## Method

1. Extract the transcript using `youtube-transcript-api`.
2. Explore transcript length, snippets, punctuation and word distribution.
3. Divide the transcript into fixed 75-word windows.
4. Generate 384-dimensional semantic embeddings using `all-MiniLM-L6-v2`.
5. Calculate cosine similarity between consecutive windows.
6. Detect potential narrative boundaries using a statistical similarity threshold.
7. Filter closely adjacent boundaries.
8. Construct narrative segments with timestamps.
9. Extract named entities using spaCy's `en_core_web_sm`.
10. Store the final segments and entities in structured JSON.

## Output

The final output contains:

- Segment ID
- Start timestamp
- End timestamp
- Word count
- Segment text
- Extracted entities and their labels

The current transcript produces **133 narrative segments**.

## Technologies

- Python
- pandas
- NumPy
- Sentence Transformers
- scikit-learn
- spaCy
- YouTube Transcript API

## Project Structure

```text
news-pipeline/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── narrative_segments.csv
│   ├── entities.csv
│   └── final_segments.json
├── requirements.txt
├── README.md
└── .gitignore