import sys
import pandas as pd
from rag import load_index, get_llm, retrieve, answer, rewrite

IDK = "i don't know"


def norm(text):
    """Lowercase, straighten curly quotes, remove all spaces: '1.e4' == '1. e4'."""
    text = str(text).lower().replace("\u2019", "'")
    return "".join(text.split())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = args[0] if args else "run"
    retrieval_only = "--retrieval-only" in sys.argv
    mode = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mode=")), "keyword")
    use_rewrite = "--rewrite" in sys.argv

    questions_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--questions=")), "eval/questions.csv")
    questions = pd.read_csv(questions_file)
    index = load_index(mode)
    llm = get_llm() if (use_rewrite or not retrieval_only) else None

    rows = []
    for q in questions.itertuples():
        search_query = rewrite(q.question, llm) if use_rewrite else None
        results = retrieve(index, q.question, mode=mode, extra_query=search_query)
        answerable = pd.notna(q.expected_name)

        # Retrieval: was the expected opening among the retrieved ones?
        hit = None
        if answerable:
            hit = any(
                doc.metadata["name"] == q.expected_name and doc.metadata["eco"] == q.expected_eco
                for doc, _ in results
            )

        # Answer: does the reply contain the expected answer (or "I don't know")?
        reply, correct = "", None
        if not retrieval_only:
            reply = answer(q.question, results, llm)
            expected = q.expected_answer if answerable else IDK
            correct = norm(expected) in norm(reply)

        best = results[0][1]
        mark = lambda x: "-" if x is None else ("Y" if x else "N")
        print(f"{q.id:>2}  {q.type:<14} retrieval {mark(hit)}  answer {mark(correct)}  best {best:.2f}  {q.question[:45]}")

        rows.append({
            "id": q.id, "type": q.type, "question": q.question, "search_query": search_query,
            "retrieval_hit": hit, "answer_correct": correct, "best_distance": round(best, 3),
            "retrieved": " | ".join(f"{d.metadata['eco']} {d.metadata['name']}" for d, _ in results),
            "reply": reply,
        })

    df = pd.DataFrame(rows)
    answerable = df[df.retrieval_hit.notna()]
    print(f"\n=== {label} (mode: {mode}, rewrite: {use_rewrite}) ===")
    print(f"Retrieval: {int(answerable.retrieval_hit.sum())} / {len(answerable)} answerable questions found the right opening")
    if not retrieval_only:
        print(f"Answers:   {int(df.answer_correct.sum())} / {len(df)} correct")
    print("\nBy type:")
    print(df.groupby("type")[["retrieval_hit", "answer_correct"]].agg(lambda s: f"{int(s.sum())}/{s.notna().sum()}").to_string())

    out = f"eval/results_{label}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved details to {out}")


if __name__ == "__main__":
    main()