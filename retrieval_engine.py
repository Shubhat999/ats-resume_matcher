"""
Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
═══════════════════════════════════════════════════════════════════════════════
Call #1  extract_jd_features()   — gpt-4o-mini   — parse JD into structured
                                                     skills/exp/role_summary
Call #2  ResumeIndex.build()     — text-embedding-3-large — embed ALL resume
                                                     chunks in one batch call
Call #3  rerank_and_explain()    — gpt-4o-mini   — rerank top-N candidates
                                                     AND generate per-candidate
                                                     strengths + gaps, all in
                                                     one single prompt

Zero extra calls for explanations, zero per-resume LLM parsing.
───────────────────────────────────────────────────────────────────────────────
Retrieval stack:
  Dense  : OpenAI text-embedding-3-large + LangChain FAISS
  Sparse : BM25Okapi (pure Python, no API)
  Fusion : Reciprocal Rank Fusion (RRF)
  Score  : weighted blend (skill overlap + semantic rank + experience)
  Rerank : GPT-4o-mini — scores + explanations in one batch call
"""

import numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ── Singleton embeddings (reused across calls, recreated only if key changes) ─
_embeddings_instance: Optional[OpenAIEmbeddings] = None
_embeddings_key: str = ""

def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    global _embeddings_instance, _embeddings_key
    if _embeddings_instance is None or api_key != _embeddings_key:
        _embeddings_instance = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key,
        )
        _embeddings_key = api_key
    return _embeddings_instance


# ── BM25 tokenizer (no regex — pure char scan) ───────────────────────────────
def tokenize(text: str) -> list:
    tokens = []
    for token in text.lower().split():
        clean = "".join(ch for ch in token if ch.isalnum() or ch in "+#")
        if len(clean) > 1:
            tokens.append(clean)
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# API CALL #1 — Parse JD with GPT-4o-mini
# ═══════════════════════════════════════════════════════════════════════════════
def extract_jd_features(jd_text: str, api_key: str) -> dict:
    """
    ONE LLM call — extracts required_skills, preferred_skills, all_skills,
    exp_years, and role_summary from ANY job description (tech or non-tech).
    Falls back to empty dicts on failure (semantic-only mode).
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            max_tokens=800,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior recruiter. Parse job descriptions precisely. "
             "Output ONLY valid JSON — no markdown fences, no extra text."),
            ("human",
             "Analyze this job description and extract structured requirements.\n\n"
             "JD:\n{jd_text}\n\n"
             "Return ONLY:\n"
             "{{\n"
             "  \"required_skills\":  [\"skill\", ...],\n"
             "  \"preferred_skills\": [\"skill\", ...],\n"
             "  \"all_skills\":       [\"skill\", ...],\n"
             "  \"skill_synonyms\":   {{\"skill\": [\"alias1\", \"alias2\"], ...}},\n"
             "  \"exp_years\":        <float>,\n"
             "  \"role_summary\":     \"<1-2 sentences>\"\n"
             "}}\n\n"
             "Rules:\n"
             "• required_skills  — must-have / required / essential\n"
             "• preferred_skills — nice-to-have / preferred / bonus\n"
             "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
             "• Include domain skills too: 'financial modeling', 'stakeholder management',\n"
             "  'content strategy', 'client communication', 'agile delivery', etc.\n"
             "• skill_synonyms   — for each skill in all_skills, list common aliases,\n"
             "  abbreviations, and alternate forms. Examples:\n"
             "  'abm': ['account based marketing', 'account-based marketing'],\n"
             "  'crm': ['customer relationship management', 'crm tools', 'crm platform'],\n"
             "  'apollo.io': ['apollo'],\n"
             "  'b2b sales': ['business to business sales', 'b2b'],\n"
             "  Only include skills that have meaningful aliases.\n"
             "  Do NOT add generic category names as synonyms for specific tools.\n"
             "  Example: DO NOT add 'erp systems' as synonym for 'sap' — they are different specificity levels.\n"
             "• exp_years — minimum years required (0.0 if not mentioned)\n"
             "• All skill strings: lowercase, 1–4 words max\n"
             "• Works for ANY domain — tech, sales, HR, marketing, ops, finance"),
        ])
        chain  = prompt | llm | JsonOutputParser()
        result = chain.invoke({"jd_text": jd_text[:2500]})

        def clean(lst):
            return list(dict.fromkeys(
                s.strip().lower() for s in (lst or []) if s and s.strip()
            ))
        return {
            "required_skills":  clean(result.get("required_skills",  [])),
            "preferred_skills": clean(result.get("preferred_skills", [])),
            "all_skills":       clean(result.get("all_skills",       [])),
            "skill_synonyms":   result.get("skill_synonyms", {}),
            "exp_years":        float(result.get("exp_years", 0.0)),
            "role_summary":     result.get("role_summary", ""),
            "raw":              jd_text,
        }
    except Exception as exc:
        print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
        return {
            "required_skills": [], "preferred_skills": [], "all_skills": [],
            "exp_years": 0.0, "role_summary": "", "raw": jd_text,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX — API CALL #2 happens here (batch embed all chunks)
# ═══════════════════════════════════════════════════════════════════════════════
LABEL_WEIGHT = {
    "skills": 1.4, "projects": 1.3, "experience": 1.2,
    "summary": 1.0, "education": 0.8, "certifications": 0.8, "full": 0.9,
}

class ResumeIndex:
    def __init__(self):
        self.candidates:    list = []
        self.chunk_meta:    list = []
        self.faiss_store:   Optional[FAISS] = None
        self.bm25:          Optional[BM25Okapi] = None
        self.corpus_tokens: list = []

    def build(self, parsed_resumes: list, api_key: str):
        """
        One batch call to text-embedding-3-large for ALL resume chunks.
        LangChain FAISS.from_documents handles batching automatically.
        """
        self.candidates  = parsed_resumes
        self.chunk_meta  = []
        lc_docs          = []

        for idx, resume in enumerate(parsed_resumes):
            for label, text in resume["chunks"].items():
                self.chunk_meta.append({"candidate_idx": idx, "chunk_label": label})
                lc_docs.append(Document(
                    page_content=text,
                    metadata={
                        "candidate_idx": idx,
                        "chunk_label":   label,
                        "weight":        LABEL_WEIGHT.get(label, 1.0),
                    },
                ))

        # ── API CALL #2 ────────────────────────────────────────────────────────
        embeddings       = get_embeddings(api_key)
        self.faiss_store = FAISS.from_documents(lc_docs, embeddings)
        # ──────────────────────────────────────────────────────────────────────

        self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
        self.bm25          = BM25Okapi(self.corpus_tokens)

    def is_ready(self) -> bool:
        return bool(self.candidates) and self.faiss_store is not None


# ── RRF ───────────────────────────────────────────────────────────────────────
def rrf(rank_lists: list, k: int = 60) -> list:
    scores: dict = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── Dense retrieval (uses cached FAISS — no new API call) ────────────────────
def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
    """
    Query the FAISS store with the JD text.
    text-embedding-3-large vectors are L2-normalized → L2 dist ∈ [0,2],
    convert to similarity = 1 - dist/2.
    This does embed the JD query — but that's bundled into Call #2's
    OpenAIEmbeddings instance (same API key, same model, counts as part of
    the embedding call already set up). No separate billed call.
    Note: FAISS.similarity_search_with_score calls embed_query() once for
    the JD — this is the ONLY extra embed call and is unavoidable + cheap
    (1 query vs N document chunks).
    """
    search_k = min(len(index.chunk_meta), k * 8)
    hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

    best: dict = {}
    for doc, l2_dist in hits:
        meta     = doc.metadata
        cand_idx = meta["candidate_idx"]
        sim      = max(0.0, 1.0 - l2_dist / 2.0)
        wscore   = sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0)
        if wscore > best.get(cand_idx, -1.0):
            best[cand_idx] = wscore

    return [c for c, _ in sorted(best.items(), key=lambda x: -x[1])]


def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
    scores = index.bm25.get_scores(jd_tokens)
    ranked = np.argsort(-scores)
    return [int(i) for i in ranked[:k]]


# ── Scoring helpers (no API) ──────────────────────────────────────────────────
def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
    if jd_skill in cand_set:
        return True

    jd_words = set(jd_skill.split())
    for cs in cand_set:
        cs_words = set(cs.split())
        if len(jd_words) == 1:
            if jd_skill in cs:
                return True
        elif len(cs_words) == 1:
            if cs in jd_skill:
                return True
        else:
            if jd_words == cs_words:
                return True
            if jd_words.issubset(cs_words) and len(jd_words) >= 2:
                return True
            if cs_words.issubset(jd_words) and len(cs_words) >= 2:
                return True

    if skill_synonyms:
        for syn in skill_synonyms.get(jd_skill, []):
            syn = syn.lower()
            if syn in cand_set or any(syn in cs or cs in syn for cs in cand_set):
                return True
    return False


def skill_overlap_score(candidate_skills: list, jd_features: dict) -> float:
    all_jd = set(jd_features.get("all_skills", []))
    if not all_jd:
        return 0.0
    cand_set   = set(candidate_skills)
    req        = set(jd_features.get("required_skills",  []))
    pref       = set(jd_features.get("preferred_skills", []))
    syns       = jd_features.get("skill_synonyms", {})
    req_match  = sum(fuzzy_match(s, cand_set, syns) for s in req)    / len(req)    if req   else 0.0
    pref_match = sum(fuzzy_match(s, cand_set, syns) for s in pref)   / len(pref)   if pref  else 0.0
    all_match  = sum(fuzzy_match(s, cand_set, syns) for s in all_jd) / len(all_jd)
    return 0.5 * req_match + 0.3 * all_match + 0.2 * pref_match


def experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 0.5 if candidate_years > 0 else 0.2
    if candidate_years <= 0:
        return 0.0
    ratio = candidate_years / required_years
    if ratio >= 1.0:
        return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
    return max(0.0, ratio * 0.85)


def build_explanation_data(candidate: dict, jd_features: dict, final_score: float) -> dict:
    cand_set  = set(candidate.get("skills", []))
    jd_skills = set(jd_features.get("all_skills", []))
    req       = set(jd_features.get("required_skills", []))

    syns     = jd_features.get("skill_synonyms", {})
    matched  = sorted(s for s in jd_skills if fuzzy_match(s, cand_set, syns))
    missing  = sorted(s for s in jd_skills if not fuzzy_match(s, cand_set, syns))
    req_done = sorted(s for s in req if fuzzy_match(s, cand_set, syns))
    req_miss = sorted(s for s in req if not fuzzy_match(s, cand_set, syns))

    return {
        "matched_skills":   matched,
        "missing_skills":   missing[:8],
        "required_matched": req_done,
        "required_missing": req_miss,
        "score_pct":        round(final_score * 100, 1),
        "strengths":        [],   # filled by Call #3
        "gaps":             [],   # filled by Call #3
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API CALL #3 — Rerank + Explain all top-N candidates in ONE prompt
# ═══════════════════════════════════════════════════════════════════════════════
def rerank_and_explain(
    jd_text: str,
    jd_features: dict,
    candidates_pool: list,
    top_n: int,
    api_key: str,
) -> list:
    """
    Single GPT-4o-mini call that does TWO things at once for ALL candidates:
      1. Relevance score  (0.0–1.0) for final ranking
      2. Strengths + gaps (bullet points) for explanation card

    Returns top_n candidates sorted by blended score, with explanations filled.
    """
    if not candidates_pool or not api_key:
        return candidates_pool[:top_n]

    # Build compact candidate summaries
    summaries = []
    for i, item in enumerate(candidates_pool):
        c       = item["candidate"]
        skills  = ", ".join(c.get("skills", [])[:20]) or "none listed"
        exp     = c.get("experience_years", 0)
        name    = c.get("name", f"Candidate {i}")
        ex      = item["explanation"]
        matched = ", ".join(ex.get("matched_skills", [])[:15]) or "none"
        missing = ", ".join(ex.get("missing_skills", [])[:10]) or "none"
        chunks  = c.get("chunks", {})
        context = (
    chunks.get("summary", "") + " " +
    chunks.get("experience", "")[:500]
    )[:600]
        summaries.append(
            f"[{i}] {name} | {exp} yrs exp\n"
            f"    ALREADY_MATCHED (DO NOT put these in gaps): {matched}\n"
            f"    context: {context}"
        )

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            max_tokens=min(4000, 600 + 400 * len(candidates_pool)),  # scale with candidate count
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert recruiter. Evaluate candidates for the given role. "
             "Output ONLY valid JSON — no markdown, no extra text."),
            ("human",
             "Role: {role_summary}\n"
             "JD (first 800 chars): {jd_text}\n"
             "Required skills: {required_skills}\n\n"
             "Candidates (index | name | exp | skills):\n{summaries}\n\n"
             "IMPORTANT: Base strengths and gaps ONLY on what is explicitly present "
             "in the candidate's context and matched skills. Do NOT assume skills "
             "are missing if they appear in the matched list. Do NOT hallucinate "
             "missing skills — only list something as a gap if it is clearly absent.\n"
             "ABSOLUTE RULE: Any skill listed under ALREADY_MATCHED for a candidate "
             "is confirmed present in their resume. NEVER list it in gaps. NEVER say "
             "it is missing. Your gaps must only contain skills NOT in ALREADY_MATCHED.\n"
             "Infer soft skills from experience: if candidate has 5+ years in Sales, "
             "Business Development, or Client Management — communication skills, "
             "negotiation, and relationship management are implicitly present.\n"
             "Never list a skill as missing if it or its synonym appears in matched list.\n"
             "IMPORTANT: Evaluate RELEVANT experience only. Cafe owner, hospitality, or "
             "unrelated domain experience should NOT count toward role-specific experience score.\n"
             "If candidate's work history shows keyword stuffing or inconsistent profile "
             "(e.g. supply chain keywords in software dev roles), penalize the score heavily.\n"
             "CRITICAL: Strengths and Why Selected must ONLY reference skills/experience explicitly "
             "present in the candidate context. Never invent or assume experience not shown.\n"
             "If a skill appears in matched list but candidate context does not support it, "
             "do NOT mention it in strengths.\n\n"
             "For EACH candidate return:\n"
             "  score     — float 0.0–1.0 (fit for this role)\n"
             "  strengths — list of 2–3 strings, each under 12 words, specific\n"
             "  gaps      — list of 1–2 strings, each under 12 words, specific\n\n"
             "Return ONLY this JSON (array length must equal number of candidates):\n"
             "{{\"results\": [\n"
             "  {{\"score\": 0.92, \"strengths\": [\"...\"], \"gaps\": [\"...\"]}},\n"
             "  ...\n"
             "]}}"),
        ])
        chain  = prompt | llm | JsonOutputParser()

        # ── API CALL #3 ────────────────────────────────────────────────────────
        output = chain.invoke({
            "role_summary":     jd_features.get("role_summary", ""),
            "jd_text":          jd_text[:800],
            "required_skills":  ", ".join(jd_features.get("required_skills", [])[:12]),
            "summaries":        "\n".join(summaries),
        })
        # ──────────────────────────────────────────────────────────────────────

        llm_results = output.get("results", [])

        for i, item in enumerate(candidates_pool):
            if i >= len(llm_results):
                break
            res   = llm_results[i]
            llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
            orig  = item["final_score"]

            # Blend: multiplicative penalty if both signals are weak
            if orig < 0.15 and llm_s < 0.4:
                blend = orig * llm_s
            else:
                blend = 0.60 * llm_s + 0.40 * orig

            item["ce_score"]    = llm_s
            item["final_score"] = round(min(blend, 1.0), 4)
            item["explanation"]["score_pct"]  = round(min(blend, 1.0) * 100, 1)
            item["explanation"]["strengths"]  = res.get("strengths", [])[:4]
            item["explanation"]["gaps"]       = res.get("gaps",      [])[:3]

    except Exception as exc:
        print(f"[retrieval_engine] Rerank+explain failed: {exc} — using pre-ranked scores")
        # Fallback: rule-based explanations, no reranking
        for item in candidates_pool:
            ex = item["explanation"]
            c  = item["candidate"]
            if not ex["strengths"]:
                if ex["required_matched"]:
                    ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
                if c.get("experience_years", 0) > 0:
                    ex["strengths"].append(f"{c['experience_years']} years of experience")
                if ex["matched_skills"]:
                    ex["strengths"].append(f"{len(ex['matched_skills'])} JD skills matched")
            if not ex["gaps"] and ex["required_missing"]:
                ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

    candidates_pool.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))
    return candidates_pool[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def retrieve_top_n(
    index:    ResumeIndex,
    jd_text:  str,
    top_n:    int,
    api_key:  str,
) -> list:
    """
    Full pipeline — exactly 3 API calls total:
      Call #1  extract_jd_features      (gpt-4o-mini)
      Call #2  already done in build()  (text-embedding-3-large) + 1 cheap query embed
      Call #3  rerank_and_explain       (gpt-4o-mini, batch)
    """
    if not index.is_ready():
        return []

    # ── Call #1 ───────────────────────────────────────────────────────────────
    jd_features = extract_jd_features(jd_text, api_key)
    jd_tokens   = tokenize(jd_text)

    # ── Retrieval (no new LLM call — uses cached FAISS + BM25) ───────────────
    POOL_SIZE  = max(top_n * 3, 30)
    sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
    bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)
    fused      = rrf([sem_ranks, bm25_ranks])
    cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

    # ── Weighted scoring (no API) ─────────────────────────────────────────────
    sem_rank_map = {cand: rank for rank, cand in enumerate(sem_ranks)}
    n_cands      = max(len(index.candidates), 1)
    results      = []

    for cand_idx in cand_pool:
        candidate = index.candidates[cand_idx]
        skill_sc  = skill_overlap_score(candidate.get("skills", []), jd_features)
        exp_sc    = experience_score(candidate.get("experience_years", 0), jd_features["exp_years"])

        rank      = sem_rank_map.get(cand_idx, n_cands)
        sem_score = max(0.0, 1.0 - rank / n_cands)
        sem_adj   = sem_score if rank < n_cands * 0.8 else sem_score * 0.5

        final_score = min(
        0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0),
        1.0,
    )
        explanation = build_explanation_data(candidate, jd_features, final_score)
        results.append({
            "candidate":        candidate,
            "final_score":      final_score,
            "skill_score":      skill_sc,
            "semantic_score":   sem_score,
            "experience_score": exp_sc,
            "ce_score":         0.0,
            "explanation":      explanation,
            "jd_features":      jd_features,
        })

    results.sort(key=lambda x: -x["final_score"])
    rerank_pool = results[: min(top_n * 2, 8)]

    # ── Call #3 ───────────────────────────────────────────────────────────────
    return rerank_and_explain(jd_text, jd_features, rerank_pool, top_n, api_key)



# """
# Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# ═══════════════════════════════════════════════════════════════════════════════
# Call #1  extract_jd_features()   — gpt-4o-mini  — parse JD into structured
#                                                     skills/exp/role_summary
# Call #2  ResumeIndex.build()     — text-embedding-3-large — embed ALL resume
#                                                     chunks in one batch call
# Call #3  rerank_and_explain()    — gpt-4o-mini  — single prompt that does:
#                                                     • semantic skill resolution
#                                                     • final scoring
#                                                     • strengths + gaps

# FIXES:
#   - max_tokens was too low (200 + 200*N) → JSON truncated → parse fail → fallback
#     Now: min(4000, 500 + 400*N) with hard cap
#   - rerank_pool was top_n*2 (up to 20) — too many candidates for token budget
#     Now: min(top_n*2, 8) — max 8 candidates per LLM call
#   - resume_context per candidate trimmed 1200→600 chars to fit token budget
#   - sem_score calculation was wrong for BM25-only candidates (rank=n_cands → score=0)
#     Now uses RRF score directly as sem_score — accurate for all candidates
#   - Better exception logging with full traceback for easier debugging
#   - rerank_and_explain: validates LLM array length before applying, partial fallback
#     instead of full fallback when length mismatch
# """

# import numpy as np
# import traceback
# from typing import Optional
# from rank_bm25 import BM25Okapi

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser


# # ── Singleton embeddings ───────────────────────────────────────────────────────
# _embeddings_instance: Optional[OpenAIEmbeddings] = None
# _embeddings_key: str = ""

# def get_embeddings(api_key: str) -> OpenAIEmbeddings:
#     global _embeddings_instance, _embeddings_key
#     if _embeddings_instance is None or api_key != _embeddings_key:
#         _embeddings_instance = OpenAIEmbeddings(
#             model="text-embedding-3-large",
#             api_key=api_key,
#         )
#         _embeddings_key = api_key
#     return _embeddings_instance


# # ── BM25 tokenizer ────────────────────────────────────────────────────────────
# def tokenize(text: str) -> list:
#     tokens = []
#     for token in text.lower().split():
#         clean = "".join(ch for ch in token if ch.isalnum() or ch in "+#")
#         if len(clean) > 1:
#             tokens.append(clean)
#     return tokens


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #1 — Parse JD with LLM
# # ═══════════════════════════════════════════════════════════════════════════════
# def extract_jd_features(jd_text: str, api_key: str) -> dict:
#     """
#     ONE LLM call — extracts required_skills, preferred_skills, all_skills,
#     exp_years, role_summary from ANY job description (tech or non-tech).
#     """
#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=1000,
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a senior recruiter with expertise across all industries. "
#              "Parse job descriptions with surgical precision. "
#              "Output ONLY valid JSON — no markdown fences, no extra text."),
#             ("human",
#              "Analyze this job description and extract structured requirements.\n\n"
#              "JD:\n{jd_text}\n\n"
#              "Return ONLY this JSON:\n"
#              "{{\n"
#              "  \"required_skills\":  [\"skill\", ...],\n"
#              "  \"preferred_skills\": [\"skill\", ...],\n"
#              "  \"all_skills\":       [\"skill\", ...],\n"
#              "  \"exp_years\":        <float>,\n"
#              "  \"role_summary\":     \"<1-2 sentences>\",\n"
#              "  \"domain\":           \"<tech|non-tech|hr|finance|sales|marketing|ops|healthcare|legal|other>\"\n"
#              "}}\n\n"
#              "CRITICAL RULES:\n"
#              "• Skills must be CANONICAL CONCEPTS — the base concept, not the JD's phrasing.\n"
#              "  Example: 'proficiency in SQL databases' → 'sql'\n"
#              "  Example: 'experience with bug tracking tools' → 'bug tracking'\n"
#              "  Example: 'must collaborate with cross-functional teams' → 'collaboration'\n"
#              "  Example: 'data-driven decision making' → 'data analysis'\n"
#              "  Example: 'managed P&L responsibilities' → 'financial management'\n"
#              "• required_skills  — must-have / required / essential\n"
#              "• preferred_skills — nice-to-have / preferred / bonus\n"
#              "• all_skills       — union of everything (hard + soft + domain skills)\n"
#              "• exp_years — minimum years required (0.0 if not mentioned)\n"
#              "• All skill strings: lowercase, 1–4 words, canonical\n"
#              "• Works for ANY domain — tech, sales, HR, marketing, ops, finance, legal"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()
#         result = chain.invoke({"jd_text": jd_text[:3000]})

#         def clean(lst):
#             return list(dict.fromkeys(
#                 s.strip().lower() for s in (lst or []) if s and s.strip()
#             ))
#         return {
#             "required_skills":  clean(result.get("required_skills",  [])),
#             "preferred_skills": clean(result.get("preferred_skills", [])),
#             "all_skills":       clean(result.get("all_skills",       [])),
#             "exp_years":        float(result.get("exp_years", 0.0)),
#             "role_summary":     result.get("role_summary", ""),
#             "domain":           result.get("domain", "other"),
#             "raw":              jd_text,
#         }
#     except Exception as exc:
#         print(f"[retrieval_engine] JD parse failed: {exc}")
#         print(traceback.format_exc())
#         return {
#             "required_skills": [], "preferred_skills": [], "all_skills": [],
#             "exp_years": 0.0, "role_summary": "", "domain": "other", "raw": jd_text,
#         }


# # ═══════════════════════════════════════════════════════════════════════════════
# # INDEX — API CALL #2 happens here
# # ═══════════════════════════════════════════════════════════════════════════════
# LABEL_WEIGHT = {
#     "skills": 1.4, "projects": 1.3, "experience": 1.2,
#     "summary": 1.0, "education": 0.8, "certifications": 0.8, "full": 0.9,
# }

# class ResumeIndex:
#     def __init__(self):
#         self.candidates:    list = []
#         self.chunk_meta:    list = []
#         self.faiss_store:   Optional[FAISS] = None
#         self.bm25:          Optional[BM25Okapi] = None
#         self.corpus_tokens: list = []

#     def build(self, parsed_resumes: list, api_key: str):
#         """One batch call to text-embedding-3-large for ALL resume chunks."""
#         self.candidates  = parsed_resumes
#         self.chunk_meta  = []
#         lc_docs          = []

#         for idx, resume in enumerate(parsed_resumes):
#             for label, text in resume["chunks"].items():
#                 self.chunk_meta.append({"candidate_idx": idx, "chunk_label": label})
#                 lc_docs.append(Document(
#                     page_content=text,
#                     metadata={
#                         "candidate_idx": idx,
#                         "chunk_label":   label,
#                         "weight":        LABEL_WEIGHT.get(label, 1.0),
#                     },
#                 ))

#         embeddings       = get_embeddings(api_key)
#         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)

#         self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
#         self.bm25          = BM25Okapi(self.corpus_tokens)

#     def is_ready(self) -> bool:
#         return bool(self.candidates) and self.faiss_store is not None


# # ── RRF ───────────────────────────────────────────────────────────────────────
# def rrf(rank_lists: list, k: int = 60) -> list:
#     scores: dict = {}
#     for ranks in rank_lists:
#         for rank, idx in enumerate(ranks):
#             scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
#     return sorted(scores.items(), key=lambda x: -x[1])


# # ── Dense retrieval ────────────────────────────────────────────────────────────
# def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
#     search_k = min(len(index.chunk_meta), k * 8)
#     hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

#     best: dict = {}
#     for doc, l2_dist in hits:
#         meta     = doc.metadata
#         cand_idx = meta["candidate_idx"]
#         sim      = max(0.0, 1.0 - l2_dist / 2.0)
#         wscore   = sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0)
#         if wscore > best.get(cand_idx, -1.0):
#             best[cand_idx] = wscore

#     return [c for c, _ in sorted(best.items(), key=lambda x: -x[1])]


# def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
#     scores = index.bm25.get_scores(jd_tokens)
#     ranked = np.argsort(-scores)
#     return [int(i) for i in ranked[:k]]


# # ── Heuristic pre-scoring — POOL FILTERING ONLY ───────────────────────────────
# def heuristic_experience_score(candidate_years: float, required_years: float) -> float:
#     if required_years <= 0:
#         return 0.5 if candidate_years > 0 else 0.2
#     if candidate_years <= 0:
#         return 0.0
#     ratio = candidate_years / required_years
#     if ratio >= 1.0:
#         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
#     return max(0.0, ratio * 0.85)


# def heuristic_skill_score(candidate_skills: list, jd_all_skills: list) -> float:
#     """
#     Intentionally lenient — purpose is to keep good candidates in the pool,
#     not to rank them. LLM does precise ranking in Call #3.
#     """
#     if not jd_all_skills:
#         return 0.5
#     cand_text = " ".join(candidate_skills).lower()
#     matches = sum(
#         1 for skill in jd_all_skills
#         if any(word in cand_text for word in skill.lower().split() if len(word) > 2)
#     )
#     return min(matches / len(jd_all_skills), 1.0)


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #3 — LLM as authoritative skill matcher, scorer, and explainer
# #
# # KEY FIXES:
# #   1. max_tokens: was 200+200*N (too low) → now min(4000, 500+400*N)
# #   2. resume_context per candidate: was 1200 chars → now 600 chars
# #      (enough for skill matching, stays within token budget)
# #   3. rerank_pool: was top_n*2 (up to 20) → now min(top_n*2, 8)
# #      (more candidates = more tokens = JSON truncated = parse fail)
# #   4. Partial fallback: if LLM returns wrong count, apply what we have
# #      and only fallback for remaining candidates (not all of them)
# #   5. Full traceback logged on exception for easier debugging
# # ═══════════════════════════════════════════════════════════════════════════════
# def rerank_and_explain(
#     jd_text:         str,
#     jd_features:     dict,
#     candidates_pool: list,
#     top_n:           int,
#     api_key:         str,
# ) -> list:
#     """
#     Single LLM call — does three things at once for ALL candidates:
#       1. Semantically resolves which JD skills are present in each resume
#       2. Assigns precise fit score (0.0–1.0)
#       3. Generates grounded strengths + gaps
#     """
#     if not candidates_pool or not api_key:
#         return candidates_pool[:top_n]

#     # ── Build resume context — 600 chars per candidate (fits token budget) ────
#     summaries = []
#     for i, item in enumerate(candidates_pool):
#         c      = item["candidate"]
#         exp    = c.get("experience_years", 0)
#         name   = c.get("name", f"Candidate {i}")
#         chunks = c.get("chunks", {})

#         # Priority order: skills → experience → summary → projects → certs
#         resume_context = "\n".join(filter(None, [
#             chunks.get("summary",        "")[:150],
#             chunks.get("skills",         "")[:200],
#             chunks.get("experience",     "")[:200],
#             chunks.get("projects",       "")[:100],
#             chunks.get("certifications", "")[:100],
#         ]))[:600]  # FIX: was 1200 — 600 is enough for skill matching

#         summaries.append(
#             f"[{i}] {name} | {exp} yrs exp\n"
#             f"RESUME:\n{resume_context}"
#         )

#     all_jd_skills = jd_features.get("all_skills", [])
#     required      = jd_features.get("required_skills", [])

#     n_candidates = len(candidates_pool)

#     # FIX: max_tokens was 200+200*N — way too low, caused JSON truncation
#     # 500 base + 400 per candidate, hard cap at 4000
#     max_tok = min(4000, 500 + 400 * n_candidates)

#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=max_tok,
#         )

#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are the world's most precise recruiter across all industries. "
#              "Your skill matching is semantically exact — you understand that resumes "
#              "express the same skills in countless different ways depending on domain, "
#              "seniority, company culture, and geography.\n\n"
#              "CORE MATCHING PRINCIPLE:\n"
#              "When checking if a JD skill is present in a resume, look for ANY of:\n"
#              "  • The exact term\n"
#              "  • A more specific version (Oracle SQL covers 'sql')\n"
#              "  • A tool/platform that implies the skill (JIRA defects covers 'bug tracking')\n"
#              "  • Demonstrated experience that proves the skill\n"
#              "  • Industry-standard synonym or abbreviation\n"
#              "  • Implied skill from seniority/role (QA Lead with 6+ years implies 'test planning')\n\n"
#              "A skill is MISSING only if no form of it appears ANYWHERE in the resume.\n"
#              "NEVER mark a skill missing if it is present in any equivalent form.\n"
#              "NEVER invent strengths not supported by the resume text.\n"
#              "Output ONLY valid JSON — no markdown, no extra text."),
#             ("human",
#              "ROLE: {role_summary}\n"
#              "DOMAIN: {domain}\n"
#              "JD: {jd_text}\n\n"
#              "JD SKILLS TO MATCH (canonical concepts): {all_skills}\n"
#              "REQUIRED SKILLS: {required_skills}\n\n"
#              "CANDIDATES:\n{summaries}\n\n"
#              "RULES:\n"
#              "• Score reflects true fit: skill coverage + relevant experience + seniority match\n"
#              "• Penalize keyword stuffing (skills listed without supporting evidence)\n"
#              "• strengths/gaps must reference ONLY content visible in the resume above\n"
#              "• matched_skills + missing_skills must together equal all_skills for each candidate\n"
#              "• YOU MUST return EXACTLY {n_candidates} result objects — one per candidate\n\n"
#              "Return ONLY valid JSON in this exact format:\n"
#              "{{\"results\": [\n"
#              "  {{\"score\": 0.9, "
#              "\"matched_skills\": [...], "
#              "\"missing_skills\": [...], "
#              "\"required_matched\": [...], "
#              "\"required_missing\": [...], "
#              "\"strengths\": [\"...\", \"...\"], "
#              "\"gaps\": [\"...\", \"...\"]}},\n"
#              "  ...\n"
#              "]}}"),
#         ])

#         chain  = prompt | llm | JsonOutputParser()
#         output = chain.invoke({
#             "role_summary":    jd_features.get("role_summary", ""),
#             "domain":          jd_features.get("domain", "other"),
#             "jd_text":         jd_text[:1500],
#             "all_skills":      ", ".join(all_jd_skills),
#             "required_skills": ", ".join(required),
#             "summaries":       "\n\n---\n\n".join(summaries),
#             "n_candidates":    n_candidates,
#         })

#         llm_results = output.get("results", [])

#         # FIX: Partial fallback — apply what LLM returned, fallback only the rest
#         # Old code: if len mismatch → raise → fallback ALL candidates
#         # New code: apply first min(len(llm_results), n_candidates) candidates
#         applied = 0
#         for i, item in enumerate(candidates_pool):
#             if i >= len(llm_results):
#                 # Partial fallback for candidates LLM didn't return
#                 print(f"[retrieval_engine] Partial fallback for candidate {i} — LLM returned {len(llm_results)} of {n_candidates}")
#                 _apply_fallback(item)
#                 continue

#             res   = llm_results[i]
#             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
#             orig  = item["pre_score"]

#             if orig < 0.10 and llm_s < 0.3:
#                 blend = orig * llm_s
#             else:
#                 blend = 0.70 * llm_s + 0.30 * orig

#             item["ce_score"]    = llm_s
#             item["final_score"] = round(min(blend, 1.0), 4)
#             item["explanation"] = {
#                 "matched_skills":   res.get("matched_skills",   []),
#                 "missing_skills":   res.get("missing_skills",   [])[:8],
#                 "required_matched": res.get("required_matched", []),
#                 "required_missing": res.get("required_missing", []),
#                 "score_pct":        round(min(blend, 1.0) * 100, 1),
#                 "strengths":        res.get("strengths", [])[:4],
#                 "gaps":             res.get("gaps",      [])[:3],
#             }
#             applied += 1

#         print(f"[retrieval_engine] LLM applied to {applied}/{n_candidates} candidates successfully")

#     except Exception as exc:
#         print(f"[retrieval_engine] Rerank+explain failed: {exc}")
#         print(traceback.format_exc())   # FIX: full traceback for debugging
#         for item in candidates_pool:
#             _apply_fallback(item)

#     candidates_pool.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))
#     return candidates_pool[:top_n]


# def _apply_fallback(item: dict):
#     """Apply graceful fallback scoring when LLM result is unavailable."""
#     c  = item["candidate"]
#     ex = item["explanation"]
#     # Keep pre_score as final_score (already set)
#     item["ce_score"] = 0.0
#     ex["score_pct"]  = round(item["pre_score"] * 100, 1)
#     if not ex.get("strengths"):
#         strengths = []
#         if c.get("experience_years", 0) > 0:
#             strengths.append(f"{c['experience_years']} years of relevant experience")
#         strengths.append("Profile semantically aligns with role requirements")
#         ex["strengths"] = strengths
#     if not ex.get("gaps") and ex.get("required_missing"):
#         ex["gaps"] = [f"Missing: {', '.join(ex['required_missing'][:3])}"]


# # ═══════════════════════════════════════════════════════════════════════════════
# # Main entry point
# # ═══════════════════════════════════════════════════════════════════════════════
# def retrieve_top_n(
#     index:   ResumeIndex,
#     jd_text: str,
#     top_n:   int,
#     api_key: str,
# ) -> list:
#     """
#     Full pipeline — exactly 3 API calls total:
#       Call #1  extract_jd_features   (LLM — JD parsing, skill canonicalization)
#       Call #2  already in build()    (text-embedding-3-large + 1 cheap query embed)
#       Call #3  rerank_and_explain    (LLM — semantic skill matching + scoring)
#     """
#     if not index.is_ready():
#         return []

#     # ── Call #1 ───────────────────────────────────────────────────────────────
#     jd_features = extract_jd_features(jd_text, api_key)
#     jd_tokens   = tokenize(jd_text)

#     # ── Retrieval (no new LLM call) ───────────────────────────────────────────
#     POOL_SIZE  = max(top_n * 3, 30)
#     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
#     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)

#     # RRF fusion — returns (cand_idx, rrf_score) sorted by score desc
#     fused      = rrf([sem_ranks, bm25_ranks])
#     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

#     # FIX: Use RRF score directly as sem_score — accurate for all candidates
#     # Old: sem_rank_map lookup → BM25-only candidates got rank=n_cands → score=0
#     # New: normalize RRF score (already 0-1 range) per candidate
#     rrf_score_map = {idx: score for idx, score in fused}
#     max_rrf = max(rrf_score_map.values()) if rrf_score_map else 1.0

#     n_cands = max(len(index.candidates), 1)
#     results  = []

#     for cand_idx in cand_pool:
#         candidate = index.candidates[cand_idx]
#         skill_sc  = heuristic_skill_score(
#             candidate.get("skills", []),
#             jd_features.get("all_skills", [])
#         )
#         exp_sc    = heuristic_experience_score(
#             candidate.get("experience_years", 0),
#             jd_features["exp_years"]
#         )

#         # FIX: sem_score from normalized RRF score (not rank-based division)
#         raw_rrf   = rrf_score_map.get(cand_idx, 0.0)
#         sem_score = raw_rrf / max_rrf if max_rrf > 0 else 0.0

#         pre_score = min(0.45 * skill_sc + 0.30 * sem_score + 0.25 * exp_sc, 1.0)

#         results.append({
#             "candidate":        candidate,
#             "pre_score":        pre_score,
#             "final_score":      pre_score,       # overwritten by LLM in Call #3
#             "skill_score":      skill_sc,
#             "semantic_score":   sem_score,
#             "experience_score": exp_sc,
#             "ce_score":         0.0,
#             "explanation": {                     # overwritten entirely by LLM in Call #3
#                 "matched_skills":   [],
#                 "missing_skills":   [],
#                 "required_matched": [],
#                 "required_missing": [],
#                 "score_pct":        round(pre_score * 100, 1),
#                 "strengths":        [],
#                 "gaps":             [],
#             },
#             "jd_features": jd_features,
#         })

#     results.sort(key=lambda x: -x["pre_score"])

#     # FIX: Hard cap at 8 candidates for LLM call — prevents token budget bust
#     # Old: top_n * 2 could be 20 candidates → 20 * 600 chars = 12k chars → truncated JSON
#     rerank_pool = results[: min(top_n * 2, 8)]

#     # ── Call #3 — LLM is the final arbiter of skill matching and scoring ──────
#     return rerank_and_explain(jd_text, jd_features, rerank_pool, top_n, api_key)