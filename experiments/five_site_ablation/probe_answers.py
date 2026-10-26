"""
Why did the RAG answer NOT PRESENT when the evidence was retrieved?

Reloads saved FAISS indexes, retrieves the same top-k chunks as compare_rag.py, then re-asks each
question with every combination of:
  prompt   current (rag_march6.build_prompt) | partial (allows partial answers, NOT PRESENT only if
           nothing relevant)
  order    original retrieval order | reversed | shuffled (fixed seed)
  model    the LLMs given with --llm_models (loaded one at a time; embeddings freed first)
Saves the raw model output (compare_rag.py does not) next to the cleaned answer.
"""
import argparse
import csv
import gc
import os
import random
import sys
import time

BASE_DIR = "/work/pi_larsonj_wit_edu/superfund/project/superfund"
sys.path.insert(0, BASE_DIR)

import torch  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from benchmark_rag import EMBEDDING_MODELS, LLM_MODELS, load_llm  # noqa: E402
from rag_march6.generate import build_prompt, strip_cot  # noqa: E402
from rag_march6.retrieve import retrieve, format_context  # noqa: E402


def partial_prompt(question, chunks, site_name):
    return f"""Extract information from EPA Superfund documents about the {site_name} site. Rules:
1. Use ONLY the sources below. Do not add outside knowledge.
2. Cite every fact as [Source N].
3. Focus on the {site_name} site and its surrounding area, including its history.
4. If the sources answer only part of the question, give what they do say and briefly note what is missing.
5. Say exactly NOT PRESENT only if no source contains anything relevant to the question.
6. Be brief (1-4 sentences).

SOURCES:
{format_context(chunks)}

Question about {site_name}: {question}
Answer:"""


PROMPTS = {"current": lambda q, c, s: build_prompt(q, c, s), "partial": partial_prompt}


def orders(chunks, seed):
    shuffled = chunks[:]
    random.Random(seed).shuffle(shuffled)
    return {"original": chunks, "reversed": chunks[::-1], "shuffled": shuffled}


def generate(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True,
                                           add_generation_prompt=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return raw, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site_name", required=True)
    ap.add_argument("--indexes", nargs="+", required=True, help="label=path to a saved FAISS index")
    ap.add_argument("--questions_file", required=True)
    ap.add_argument("--embed_model", default="e5-mistral-7b", choices=list(EMBEDDING_MODELS))
    ap.add_argument("--llm_models", nargs="+", default=["gemma4-12b"], choices=list(LLM_MODELS))
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.questions_file) as f:
        questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # 1. Retrieval for every index and question, then free the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELS[args.embed_model],
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"batch_size": 8, "normalize_embeddings": True},
    )
    retrieved = {}
    for spec in args.indexes:
        label, path = spec.split("=", 1)
        index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        for q_id, q in enumerate(questions):
            retrieved[(label, q_id)] = retrieve(index, q, top_k=args.top_k)
        print(f"retrieved from {label}: {path}", flush=True)
        del index
    del embeddings
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Generation: model x index x question x prompt x order
    rows = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    for llm in args.llm_models:
        try:
            model, tokenizer, load_sec = load_llm(llm)
        except Exception as e:  # e.g. out of GPU memory for the larger model
            print(f"could not load {llm}: {e}", flush=True)
            continue
        print(f"{llm} loaded in {load_sec:.0f}s", flush=True)
        for (label, q_id), chunks in retrieved.items():
            for order_name, ordered in orders(chunks, args.seed + q_id).items():
                for prompt_name, make in PROMPTS.items():
                    raw, sec = generate(model, tokenizer, make(questions[q_id], ordered, args.site_name),
                                        args.max_new_tokens)
                    text = strip_cot(raw)
                    rows.append({
                        "llm_model": llm, "index": label, "question_id": q_id, "question": questions[q_id],
                        "prompt": prompt_name, "order": order_name,
                        "not_present": text.strip().upper().startswith("NOT PRESENT"),
                        "answer": text, "raw_response": raw, "gen_time_sec": f"{sec:.2f}",
                        "sources_in_order": " | ".join(f"{c.pdf_filename}:p{c.page}" for c in ordered),
                    })
                    print(f"  {llm} {label} Q{q_id} {prompt_name}/{order_name}: {text[:90]!r}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        with open(args.output, "w", newline="", encoding="utf-8") as f:  # write after each model
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # 3. Summary: NOT PRESENT count per condition
    md = [f"# Answer probe: {args.site_name}", "",
          f"Same top-{args.top_k} chunks per index and question; only the prompt, source order and model change.",
          "", "## NOT PRESENT answers (out of questions x indexes)", "",
          "| Model | Index | Prompt | Original order | Reversed | Shuffled |", "|---|---|---|---:|---:|---:|"]
    for llm in dict.fromkeys(r["llm_model"] for r in rows):
        for label in dict.fromkeys(r["index"] for r in rows):
            for p in PROMPTS:
                cell = []
                for o in ("original", "reversed", "shuffled"):
                    sel = [r for r in rows if (r["llm_model"], r["index"], r["prompt"], r["order"]) == (llm, label, p, o)]
                    cell.append(f"{sum(r['not_present'] for r in sel)} / {len(sel)}")
                md.append(f"| {llm} | {label} | {p} | " + " | ".join(cell) + " |")
    md += ["", "## Answers", ""]
    for q_id, q in enumerate(questions):
        md += [f"### Q{q_id}. {q}", ""]
        for r in rows:
            if r["question_id"] == q_id:
                md.append(f"- **{r['llm_model']} / {r['index']} / {r['prompt']} / {r['order']}**: "
                          + r["answer"].replace("\n", " "))
        md.append("")
    with open(os.path.splitext(args.output)[0] + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:30]))
    print("done")


if __name__ == "__main__":
    main()
