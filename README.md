# Medical Assistant: RAG over the Merck Manual

A retrieval-augmented generation (RAG) system that answers clinical questions using passages retrieved from *The Merck Manual of Diagnosis & Therapy* (4,000+ pages) rather than an LLM's general training data.

## Problem
Clinicians face information overload. A general-purpose LLM can answer medical questions fluently, but its answers aren't tied to a trusted source and can't be audited. The goal was an assistant that answers **only from a vetted reference**.

## Approach
| Step | What I did |
|---|---|
| Baseline | Queried Mistral-7B-Instruct (quantized GGUF, `llama-cpp-python`) directly on 5 clinical questions |
| Prompt engineering | Added a system prompt to constrain structure and tone |
| Ingestion | Loaded the PDF with PyMuPDF; split into 512-token chunks (tiktoken `cl100k_base`) |
| Embedding + storage | Embedded chunks with `thenlper/gte-large`; stored in a Chroma vector database |
| Retrieval + generation | Retrieved the top-k passages per question and injected them as context with a grounding-only system prompt |
| Tuning | Compared `temperature`, `top_p`, `top_k`, and number of retrieved chunks |
| Evaluation | LLM-as-a-judge scoring of **groundedness** and **relevance** (1–5) |

## Results
LLM-as-a-judge scores (1–5) for the five test questions:

| Query | Groundedness | Relevance |
|---|---|---|
| Sepsis protocol | 5 | 5 |
| Appendicitis | 5 | 5 |
| Patchy hair loss | 5 | 5 |
| Traumatic brain injury | 5 | 5 |
| Leg fracture | 5 | 5 |

**Key finding:** without retrieval, the model gave fluent but generic answers from its training data. With RAG, answers drew on specific manual content (for example, empiric antibiotic regimens for septic shock, and corticosteroid and minoxidil treatment for alopecia areata), and the judge's explanations trace each answer back to the retrieved passages.

**Caveat:** the same model generated and judged the answers, so uniformly perfect scores are likely optimistic (see limitations).

## Limitations & next steps
- The same model generates and judges answers, so scores may be optimistic. An independent judge model or clinician review would be more reliable.
- Try a biomedical embedding model and larger chunk overlap for sequential clinical instructions.
- Not for clinical use. This is a prototype for research and education.

## How to run
1. Open the notebook in Google Colab. A **GPU runtime** (T4 or better) is strongly recommended; the notebook includes a CPU fallback, but on CPU each RAG query takes several minutes.
2. Run the install cells, restart the runtime, then run all cells.
3. Place `medical_diagnosis_manual.pdf` in the working directory. The manual is copyrighted and is **not** included in this repo.

## Tools
Python · LangChain · ChromaDB · Sentence-Transformers · Hugging Face Hub · llama-cpp-python · tiktoken · PyMuPDF
