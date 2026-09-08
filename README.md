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

This project addresses the problem using an **unsupervised semantic narrative segmentation approach**.

Instead of requiring manually labelled story boundaries, the system:

1. Extracts the transcript from a YouTube video.
2. Preprocesses and explores the transcript.
3. Divides the transcript into fixed-size semantic windows.
4. Converts each window into a sentence embedding.
5. Calculates semantic similarity between consecutive windows.
6. Detects potential narrative boundaries from significant changes in similarity.
7. Constructs timestamped narrative segments.
8. Assigns semantic topic categories to the segments.
9. Extracts named entities using Named Entity Recognition (NER).
10. Extracts important keywords.
11. Produces structured JSON output.
12. Displays the results through a Streamlit web application.

---

# 🎯 Problem Statement

Given a long YouTube news video containing noisy ASR-generated subtitles:

> **Automatically identify meaningful narrative/story boundaries and extract useful structured information from each story without relying on manually labelled segmentation data.**

The system is designed for long-form news content where multiple topics can occur within the same transcript.

---

# 💡 Core Idea

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

The semantic content has changed substantially.

The system uses these semantic changes as signals for possible narrative boundaries.

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
                         │ Similarity            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Boundary Detection   │
                         │                      │
                         │ Statistical          │
                         │ Thresholding         │
                         │ Local Minima         │
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
                         │ Structured JSON      │
                         │ Output               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Application│
                         └──────────────────────┘

                         