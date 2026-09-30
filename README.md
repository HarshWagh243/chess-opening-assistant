# ♟️ Chess Opening Assistant

Ask about any chess opening in plain English, even with typos or a young student's wording, and get the moves and ECO code, with the source shown for every answer. It answers **only** from the [Lichess openings database](https://github.com/lichess-org/chess-openings) (3,815 openings) and says "I don't know" instead of guessing.

**Live demo:** [add your Streamlit link here]

I'm a chess coach, and I built this to learn retrieval-augmented generation (RAG) properly: build a baseline, measure it, and improve it one experiment at a time.

## How it works

```
question ──► LLM rewrites it into names + moves ("king on move 2" → "Ke2")
         ──► keyword search (BM25) on the original AND the rewrite
         ──► merge both result lists (Reciprocal Rank Fusion) → top 4 openings
         ──► LLM answers using ONLY those 4, citing [ECO, name] + moves
```

- **One opening = one document.** Each record is short and self-contained, so there's no chunking.
- **Grounded answers.** The prompt forbids the model's own chess knowledge. If the answer isn't in the retrieved openings, it replies "I don't know".
- **The same code runs the app and the evaluation** (`rag.py`), so the scores measure what users actually get.

## Results

I scored **retrieval** (did search find the right opening?) and **answers** (was the reply correct?) separately, so each failure can be traced to search or to the model.

| Experiment | Retrieval (/17) | Answers (/20) |
| --- | --- | --- |
| Baseline: vector search (OpenAI embeddings + FAISS) | 5 | 8 |
| Keyword search (BM25) + stopwords | 14 | 16 |
| + LLM query rewriting | 15–16 | 17 |
| + Stricter answer prompt | 15–17 | **18–20** |
| *Tried and rejected:* hybrid vector + keyword | 13 | – |
| *Tried and rejected:* few-shot rewrite prompt | 14–15 | – |

Ranges are across repeated runs. The rewrite step is non-deterministic, so I report ranges rather than the best run.

### Held-out test

The development set above was used for tuning, so its scores are optimistic. I wrote **10 new questions in the voice of 5–10-year-old students** ("carro can moves pls", "what if i move my horsey out first to f3") and ran them **once**, without tuning:

| | Retrieval | Answers |
| --- | --- | --- |
| Held-out (10 unseen questions) | **4/7** | **6/10** |

The script reported 7/10. On review, one "pass" described a different opening whose moves happen to start the same way, so the real score is 6/10.

## What I learned

- **Vector search lost badly on this data** (5/17 vs 14/17 for keyword). Opening records are short and structured: exact names, codes and move notation. Embeddings treat `f4` and `d4` as near-identical, and match "check" instead of "checkmate".
- **Hybrid search didn't help here** (13/17). Its vector half added no new hits and pushed out one correct result. I chose the simpler, faster, free option.
- **A stopword bug:** "moves of **the** Ruy Lopez" matched "Zukertort Opening: **The** Potato", because "the" is rare in opening names, so BM25 weighted it heavily.
- **Query rewriting helps and hurts.** It fixed the description questions but made others unstable. Result-merging (RRF) rewards results both lists agree on, so a drifting rewrite can outvote the correct direct match.
- **Few-shot examples were drawn from openings outside the test set** to avoid data leakage.
- **Failures are split into search failures and model failures.** After the prompt change, every remaining wrong answer came from search, not from the model.

## Known limitations

1. **Keyword search ignores move order.** For "Nf3 first", it ranked the Petrov's Defense (which contains Nf3 twice) above the Zukertort Opening (1. Nf3).
2. **Apostrophes:** "queens" doesn't match "Queen's", because they're split into different words.
3. **The rewrite can substitute a different opening** ("Queen's Gambit" → "Queen's Gambit Declined") or produce generic words.
4. **Name coincidences:** "What does Magnus Carlsen play?" returned openings *named* "Carlsen Variation".
5. **The answer check is a text match**, so it can pass answers that are really longer variations.

## Future work

- Exact move-sequence lookup for move questions, routed separately from text search
- Normalise apostrophes before tokenising
- Weighted fusion that trusts the original question more than the rewrite
- LLM-as-judge answer grading
- Add my own lesson notes (free text), where vector or hybrid search should start to pay off

## Run it locally

```bash
git clone https://github.com/HarshWagh243/chess-opening-assistant.git
cd chess-opening-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env
streamlit run app.py
```

Evaluate:

```bash
python evaluate.py dev --rewrite                                   # 20 development questions
python evaluate.py holdout --rewrite --questions=eval/holdout.csv  # 10 held-out questions
python build_index.py                                              # only needed for --mode=vector or hybrid
```

## Tech stack

Python · Streamlit · OpenAI API (gpt-5.4-mini, text-embedding-3-small) · LangChain · BM25 (rank_bm25) · FAISS · pandas

## Credits

Opening data: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings), CC0 public domain.
