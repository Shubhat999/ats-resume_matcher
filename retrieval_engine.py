# """
# Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# ═══════════════════════════════════════════════════════════════════════════════
# Call #1  extract_jd_features()   — gpt-4o-mini   — parse JD into structured
#                                                      skills/exp/role_summary
# Call #2  ResumeIndex.build()     — text-embedding-3-large — embed ALL resume
#                                                      chunks in one batch call
# Call #3  rerank_and_explain()    — gpt-4o-mini   — rerank top-N candidates
#                                                      AND generate per-candidate
#                                                      strengths + gaps, all in
#                                                      one single prompt

# Zero extra calls for explanations, zero per-resume LLM parsing.
# ───────────────────────────────────────────────────────────────────────────────
# Retrieval stack:
#   Dense  : OpenAI text-embedding-3-large + LangChain FAISS
#   Sparse : BM25Okapi (pure Python, no API)
#   Fusion : Reciprocal Rank Fusion (RRF)
#   Score  : weighted blend (skill overlap + semantic rank + experience)
#   Rerank : GPT-4o-mini — scores + explanations in one batch call

# HALLUCINATION FIXES (v2):
#   1. fuzzy_match() — tightened: no more single-char substring matches;
#      requires full word boundary containment for single-word skills.
#   2. skill_overlap_score() — "matched" requires ≥2 skills OR a required skill;
#      avoids a single noise token inflating the score.
#   3. rerank_and_explain() — ALREADY_MATCHED is now recomputed with strict match
#      before being sent to the LLM, so the LLM doesn't hallucinate strengths
#      from noise tokens.
#   4. Score blending — LLM cannot raise a low base score beyond a ceiling that
#      scales with the base; prevents a weak candidate scoring 79% via LLM alone.
#   5. Domain coherence check — if semantic similarity is below a threshold AND
#      skill overlap is also very low, apply a domain mismatch penalty before
#      passing to the LLM.
# """

# import numpy as np
# from typing import Optional
# from rank_bm25 import BM25Okapi

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser


# # ── Singleton embeddings ──────────────────────────────────────────────────────
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
# # API CALL #1 — Parse JD with GPT-4o-mini
# # ═══════════════════════════════════════════════════════════════════════════════
# def extract_jd_features(jd_text: str, api_key: str) -> dict:
#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=800,
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a senior recruiter. Parse job descriptions precisely. "
#              "Output ONLY valid JSON — no markdown fences, no extra text."),
#             ("human",
#              "Analyze this job description and extract structured requirements.\n\n"
#              "JD:\n{jd_text}\n\n"
#              "Return ONLY:\n"
#              "{{\n"
#              "  \"required_skills\":  [\"skill\", ...],\n"
#              "  \"preferred_skills\": [\"skill\", ...],\n"
#              "  \"all_skills\":       [\"skill\", ...],\n"
#              "  \"skill_synonyms\":   {{\"skill\": [\"alias1\", \"alias2\"], ...}},\n"
#              "  \"exp_years\":        <float>,\n"
#              "  \"role_summary\":     \"<1-2 sentences>\",\n"
#              "  \"domain\":           \"<primary domain, e.g. software engineering, sales, logistics, marketing>\"\n"
#              "}}\n\n"
#              "Rules:\n"
#              "• required_skills  — must-have / required / essential\n"
#              "• preferred_skills — nice-to-have / preferred / bonus\n"
#              "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
#              "• Include domain skills too: 'financial modeling', 'stakeholder management',\n"
#              "  'content strategy', 'client communication', 'agile delivery', etc.\n"
#              "• skill_synonyms   — for each skill in all_skills, list common aliases.\n"
#              "  Only include skills that have meaningful aliases.\n"
#              "  DO NOT add generic category names as synonyms for specific tools.\n"
#              "• exp_years — minimum years required (0.0 if not mentioned)\n"
#              "• domain    — single lowercase string describing the primary job domain\n"
#              "• All skill strings: lowercase, 1–4 words max\n"
#              "• Works for ANY domain — tech, sales, HR, marketing, ops, finance, logistics"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()
#         result = chain.invoke({"jd_text": jd_text[:2500]})

#         def clean(lst):
#             return list(dict.fromkeys(
#                 s.strip().lower() for s in (lst or []) if s and s.strip()
#             ))
#         return {
#             "required_skills":  clean(result.get("required_skills",  [])),
#             "preferred_skills": clean(result.get("preferred_skills", [])),
#             "all_skills":       clean(result.get("all_skills",       [])),
#             "skill_synonyms":   result.get("skill_synonyms", {}),
#             "exp_years":        float(result.get("exp_years", 0.0)),
#             "role_summary":     result.get("role_summary", ""),
#             "domain":           result.get("domain", "").lower().strip(),
#             "raw":              jd_text,
#         }
#     except Exception as exc:
#         print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
#         return {
#             "required_skills": [], "preferred_skills": [], "all_skills": [],
#             "exp_years": 0.0, "role_summary": "", "domain": "", "raw": jd_text,
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

#         # ── API CALL #2 ────────────────────────────────────────────────────────
#         embeddings       = get_embeddings(api_key)
#         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)
#         # ──────────────────────────────────────────────────────────────────────

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


# # ── Dense retrieval ───────────────────────────────────────────────────────────
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


# # ── FIX #1: Stricter fuzzy match ─────────────────────────────────────────────
# def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
#     """
#     Tighter matching rules to avoid false positives:

#     - Exact match always wins.
#     - Single-word JD skill: must appear as a WHOLE WORD in a candidate skill
#       (not just a substring). E.g. "sorting" will NOT match "address labeling".
#     - Multi-word JD skill: word-set overlap only if ≥2 words match as a proper
#       subset, preventing accidental phrase collisions.
#     - Synonym expansion is still supported but uses the same rules.
#     """
#     if jd_skill in cand_set:
#         return True

#     jd_words = set(jd_skill.split())

#     for cs in cand_set:
#         cs_words = set(cs.split())

#         if len(jd_words) == 1:
#             # ── FIX: whole-word containment, not substring ────────────────────
#             # e.g. jd="sorting" must match "mail sorting" but NOT "addressing"
#             jd_word = jd_skill
#             if jd_word in cs_words:
#                 return True
#             # Allow compound-word match: "javascript" in "javascript developer"
#             if cs.startswith(jd_word + " ") or cs.endswith(" " + jd_word):
#                 return True

#         elif len(cs_words) == 1:
#             cs_word = cs
#             if cs_word in jd_words:
#                 return True

#         else:
#             # Multi-word: require FULL word-set equality or proper 2-word subset
#             if jd_words == cs_words:
#                 return True
#             if len(jd_words) >= 2 and jd_words.issubset(cs_words):
#                 return True
#             if len(cs_words) >= 2 and cs_words.issubset(jd_words):
#                 return True

#     # Synonym expansion with same strict rules
#     if skill_synonyms:
#         for syn in skill_synonyms.get(jd_skill, []):
#             syn = syn.lower().strip()
#             syn_words = set(syn.split())
#             if syn in cand_set:
#                 return True
#             for cs in cand_set:
#                 cs_words = set(cs.split())
#                 if len(syn_words) == 1:
#                     if syn in cs_words:
#                         return True
#                 else:
#                     if syn_words == cs_words or syn_words.issubset(cs_words):
#                         return True

#     return False


# # ── FIX #2: Minimum evidence threshold in skill_overlap_score ────────────────
# def skill_overlap_score(candidate_skills: list, jd_features: dict) -> float:
#     """
#     Returns 0.0 if fewer than MIN_MATCH skills match, to prevent a single
#     noise token (e.g. a bigram from the candidate's name) from inflating the
#     score. The LLM reranker is the right place to rescue edge cases.
#     """
#     MIN_MATCH = 2  # at least 2 skills must match before we award points

#     all_jd = set(jd_features.get("all_skills", []))
#     if not all_jd:
#         return 0.0

#     cand_set  = set(candidate_skills)
#     req       = set(jd_features.get("required_skills",  []))
#     pref      = set(jd_features.get("preferred_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     all_match_list  = [s for s in all_jd if fuzzy_match(s, cand_set, syns)]
#     req_match_list  = [s for s in req    if fuzzy_match(s, cand_set, syns)]
#     pref_match_list = [s for s in pref   if fuzzy_match(s, cand_set, syns)]

#     # ── FIX: require minimum evidence ────────────────────────────────────────
#     total_matched = len(all_match_list)
#     if total_matched < MIN_MATCH:
#         # Give a tiny partial credit if exactly 1 required skill matched,
#         # but never more than 0.10 — the LLM can override upward if justified
#         if req_match_list:
#             return 0.08
#         return 0.0

#     req_ratio  = len(req_match_list)  / len(req)  if req   else 0.0
#     pref_ratio = len(pref_match_list) / len(pref) if pref  else 0.0
#     all_ratio  = len(all_match_list)  / len(all_jd)

#     return 0.5 * req_ratio + 0.3 * all_ratio + 0.2 * pref_ratio


# def experience_score(candidate_years: float, required_years: float) -> float:
#     if required_years <= 0:
#         return 0.5 if candidate_years > 0 else 0.2
#     if candidate_years <= 0:
#         return 0.0
#     ratio = candidate_years / required_years
#     if ratio >= 1.0:
#         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
#     return max(0.0, ratio * 0.85)


# # ── FIX #3: Recompute matched/missing with strict fuzzy before sending to LLM ─
# def build_explanation_data(candidate: dict, jd_features: dict, final_score: float) -> dict:
#     cand_set  = set(candidate.get("skills", []))
#     jd_skills = set(jd_features.get("all_skills", []))
#     req       = set(jd_features.get("required_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     # Use the same strict fuzzy_match — so the LLM sees honest matched/missing
#     matched  = sorted(s for s in jd_skills if fuzzy_match(s, cand_set, syns))
#     missing  = sorted(s for s in jd_skills if not fuzzy_match(s, cand_set, syns))
#     req_done = sorted(s for s in req if fuzzy_match(s, cand_set, syns))
#     req_miss = sorted(s for s in req if not fuzzy_match(s, cand_set, syns))

#     return {
#         "matched_skills":   matched,
#         "missing_skills":   missing[:8],
#         "required_matched": req_done,
#         "required_missing": req_miss,
#         "score_pct":        round(final_score * 100, 1),
#         "strengths":        [],
#         "gaps":             [],
#     }


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #3 — Rerank + Explain all top-N candidates in ONE prompt
# # ═══════════════════════════════════════════════════════════════════════════════
# def rerank_and_explain(
#     jd_text: str,
#     jd_features: dict,
#     candidates_pool: list,
#     top_n: int,
#     api_key: str,
# ) -> list:
#     if not candidates_pool or not api_key:
#         return candidates_pool[:top_n]

#     summaries = []
#     for i, item in enumerate(candidates_pool):
#         c       = item["candidate"]
#         exp     = c.get("experience_years", 0)
#         name    = c.get("name", f"Candidate {i}")
#         ex      = item["explanation"]
#         # ── FIX #3: send strictly-matched skills — no noise tokens ────────────
#         matched = ", ".join(ex.get("matched_skills", [])[:15]) or "none"
#         missing = ", ".join(ex.get("missing_skills", [])[:10]) or "none"
#         chunks  = c.get("chunks", {})
#         context = (
#             chunks.get("summary", "") + " " +
#             chunks.get("experience", "")[:500]
#         )[:600]
#         summaries.append(
#             f"[{i}] {name} | {exp} yrs exp\n"
#             f"    CONFIRMED_PRESENT (verified from resume, never list in gaps): {matched}\n"
#             f"    CONFIRMED_MISSING: {missing}\n"
#             f"    context: {context}"
#         )

#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=min(4000, 600 + 400 * len(candidates_pool)),
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are an expert recruiter. Evaluate candidates strictly based on "
#              "evidence in their resume. Never invent or assume skills not shown. "
#              "Output ONLY valid JSON — no markdown, no extra text."),
#             ("human",
#              "Role: {role_summary}\n"
#              "Domain: {domain}\n"
#              "JD (first 800 chars): {jd_text}\n"
#              "Required skills: {required_skills}\n\n"
#              "Candidates:\n{summaries}\n\n"
#              "CRITICAL RULES:\n"
#              "1. CONFIRMED_PRESENT skills are verified — they ARE in the resume. "
#              "   Do NOT list them in gaps. Never say they are missing.\n"
#              "2. CONFIRMED_MISSING skills are verified absent — list relevant ones as gaps.\n"
#              "3. Strengths must reference ONLY evidence visible in the candidate context. "
#              "   Never invent experience. If context is thin, say so in a gap instead.\n"
#              "4. Domain coherence: if the candidate's background is clearly a different domain "
#              "   (e.g. sales/marketing candidate for logistics role, or hospitality for software), "
#              "   score must reflect that mismatch — do not award high scores for transferable "
#              "   soft skills alone without hard-skill evidence.\n"
#              "5. Experience relevance: only count experience from the same or closely adjacent "
#              "   domain. Unrelated work (e.g. cafe owner, delivery driver for a software role) "
#              "   does not count as relevant experience.\n"
#              "6. Keyword stuffing: if resume has many keywords but context shows unrelated roles, "
#              "   penalise heavily (score ≤ 0.3).\n\n"
#              "For EACH candidate return:\n"
#              "  score     — float 0.0–1.0 (genuine fit for this specific role)\n"
#              "  strengths — 2–3 strings, each under 12 words, grounded in context\n"
#              "  gaps      — 1–2 strings, each under 12 words, specific missing items\n\n"
#              "Return ONLY this JSON (array length must equal number of candidates):\n"
#              "{{\"results\": [\n"
#              "  {{\"score\": 0.92, \"strengths\": [\"...\"], \"gaps\": [\"...\"]}},\n"
#              "  ...\n"
#              "]}}"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()

#         # ── API CALL #3 ────────────────────────────────────────────────────────
#         output = chain.invoke({
#             "role_summary":    jd_features.get("role_summary", ""),
#             "domain":          jd_features.get("domain", "unspecified"),
#             "jd_text":         jd_text[:800],
#             "required_skills": ", ".join(jd_features.get("required_skills", [])[:12]),
#             "summaries":       "\n".join(summaries),
#         })
#         # ──────────────────────────────────────────────────────────────────────

#         llm_results = output.get("results", [])

#         for i, item in enumerate(candidates_pool):
#             if i >= len(llm_results):
#                 break
#             res   = llm_results[i]
#             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
#             orig  = item["final_score"]

#             # ── FIX #4: Constrained score blending ───────────────────────────
#             # LLM can raise a weak base score, but not beyond a ceiling that
#             # scales with the base. Prevents 0.15 base → 0.79 final.
#             #
#             # Ceiling formula: base_score + max_lift
#             #   max_lift = 0.25 if base < 0.30 (capped rescue)
#             #   max_lift = 0.35 if 0.30 ≤ base < 0.55 (moderate lift)
#             #   max_lift = 0.50 if base ≥ 0.55 (good candidate, trust LLM more)
#             #
#             # If both are weak → multiplicative penalty (unchanged)
#             if orig < 0.15 and llm_s < 0.4:
#                 blend = orig * llm_s
#             else:
#                 if orig < 0.30:
#                     max_lift = 0.25
#                 elif orig < 0.55:
#                     max_lift = 0.35
#                 else:
#                     max_lift = 0.50

#                 raw_blend = 0.55 * llm_s + 0.45 * orig
#                 ceiling   = min(orig + max_lift, 1.0)
#                 blend     = min(raw_blend, ceiling)

#             item["ce_score"]    = llm_s
#             item["final_score"] = round(min(blend, 1.0), 4)
#             item["explanation"]["score_pct"]  = round(min(blend, 1.0) * 100, 1)
#             item["explanation"]["strengths"]  = res.get("strengths", [])[:4]
#             item["explanation"]["gaps"]       = res.get("gaps",      [])[:3]

#     except Exception as exc:
#         print(f"[retrieval_engine] Rerank+explain failed: {exc} — using pre-ranked scores")
#         for item in candidates_pool:
#             ex = item["explanation"]
#             c  = item["candidate"]
#             if not ex["strengths"]:
#                 if ex["required_matched"]:
#                     ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
#                 if c.get("experience_years", 0) > 0:
#                     ex["strengths"].append(f"{c['experience_years']} years of experience")
#                 if ex["matched_skills"]:
#                     ex["strengths"].append(f"{len(ex['matched_skills'])} JD skills matched")
#             if not ex["gaps"] and ex["required_missing"]:
#                 ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

#     candidates_pool.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))
#     return candidates_pool[:top_n]


# # ── FIX #5: Domain coherence pre-filter ──────────────────────────────────────
# def _apply_domain_penalty(results: list, jd_features: dict) -> list:
#     """
#     Before sending to the LLM reranker, apply a soft domain-mismatch penalty.
#     If a candidate has very low skill overlap AND very low semantic score,
#     cap their final_score at 0.35 so they can't leapfrog real candidates
#     purely via LLM charity scoring.

#     This is heuristic — the LLM can still score them low, but this prevents
#     the base score from being deceptively high before Call #3.
#     """
#     DOMAIN_PENALTY_THRESHOLD_SKILL    = 0.10
#     DOMAIN_PENALTY_THRESHOLD_SEMANTIC = 0.25
#     DOMAIN_PENALTY_CAP                = 0.35

#     for item in results:
#         skill_sc   = item.get("skill_score", 0.0)
#         sem_sc     = item.get("semantic_score", 0.0)
#         if skill_sc < DOMAIN_PENALTY_THRESHOLD_SKILL and sem_sc < DOMAIN_PENALTY_THRESHOLD_SEMANTIC:
#             item["final_score"] = min(item["final_score"], DOMAIN_PENALTY_CAP)
#             item["explanation"]["score_pct"] = round(min(item["final_score"], DOMAIN_PENALTY_CAP) * 100, 1)
#             # Tag so the LLM knows
#             item["_domain_penalised"] = True

#     return results


# # ═══════════════════════════════════════════════════════════════════════════════
# # Main entry point
# # ═══════════════════════════════════════════════════════════════════════════════
# def retrieve_top_n(
#     index:    "ResumeIndex",
#     jd_text:  str,
#     top_n:    int,
#     api_key:  str,
# ) -> list:
#     if not index.is_ready():
#         return []

#     # ── Call #1 ───────────────────────────────────────────────────────────────
#     jd_features = extract_jd_features(jd_text, api_key)
#     jd_tokens   = tokenize(jd_text)

#     # ── Retrieval ─────────────────────────────────────────────────────────────
#     POOL_SIZE  = max(top_n * 3, 30)
#     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
#     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)
#     fused      = rrf([sem_ranks, bm25_ranks])
#     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

#     # ── Weighted scoring ──────────────────────────────────────────────────────
#     sem_rank_map = {cand: rank for rank, cand in enumerate(sem_ranks)}
#     n_cands      = max(len(index.candidates), 1)
#     results      = []

#     for cand_idx in cand_pool:
#         candidate = index.candidates[cand_idx]
#         skill_sc  = skill_overlap_score(candidate.get("skills", []), jd_features)
#         exp_sc    = experience_score(candidate.get("experience_years", 0), jd_features["exp_years"])

#         rank      = sem_rank_map.get(cand_idx, n_cands)
#         sem_score = max(0.0, 1.0 - rank / n_cands)
#         sem_adj   = sem_score if rank < n_cands * 0.8 else sem_score * 0.5

#         final_score = min(
#             0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0),
#             1.0,
#         )
#         explanation = build_explanation_data(candidate, jd_features, final_score)
#         results.append({
#             "candidate":        candidate,
#             "final_score":      final_score,
#             "skill_score":      skill_sc,
#             "semantic_score":   sem_score,
#             "experience_score": exp_sc,
#             "ce_score":         0.0,
#             "explanation":      explanation,
#             "jd_features":      jd_features,
#         })

#     results.sort(key=lambda x: -x["final_score"])

#     # ── FIX #5: Domain penalty before LLM sees them ───────────────────────────
#     rerank_pool = results[: min(top_n * 2, 8)]
#     rerank_pool = _apply_domain_penalty(rerank_pool, jd_features)

#     # ── Call #3 ───────────────────────────────────────────────────────────────
#     return rerank_and_explain(jd_text, jd_features, rerank_pool, top_n, api_key)



# # """
# # Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# # ═══════════════════════════════════════════════════════════════════════════════
# # Call #1  extract_jd_features()   — gpt-4o-mini  — parse JD into structured
# #                                                     skills/exp/role_summary
# # Call #2  ResumeIndex.build()     — text-embedding-3-large — embed ALL resume
# #                                                     chunks in one batch call
# # Call #3  rerank_and_explain()    — gpt-4o-mini  — single prompt that does:
# #                                                     • semantic skill resolution
# #                                                     • final scoring
# #                                                     • strengths + gaps

# # FIXES:
# #   - max_tokens was too low (200 + 200*N) → JSON truncated → parse fail → fallback
# #     Now: min(4000, 500 + 400*N) with hard cap
# #   - rerank_pool was top_n*2 (up to 20) — too many candidates for token budget
# #     Now: min(top_n*2, 8) — max 8 candidates per LLM call
# #   - resume_context per candidate trimmed 1200→600 chars to fit token budget
# #   - sem_score calculation was wrong for BM25-only candidates (rank=n_cands → score=0)
# #     Now uses RRF score directly as sem_score — accurate for all candidates
# #   - Better exception logging with full traceback for easier debugging
# #   - rerank_and_explain: validates LLM array length before applying, partial fallback
# #     instead of full fallback when length mismatch
# # """

# # import numpy as np
# # import traceback
# # from typing import Optional
# # from rank_bm25 import BM25Okapi

# # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# # from langchain_community.vectorstores import FAISS
# # from langchain_core.documents import Document
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import JsonOutputParser


# # # ── Singleton embeddings ───────────────────────────────────────────────────────
# # _embeddings_instance: Optional[OpenAIEmbeddings] = None
# # _embeddings_key: str = ""

# # def get_embeddings(api_key: str) -> OpenAIEmbeddings:
# #     global _embeddings_instance, _embeddings_key
# #     if _embeddings_instance is None or api_key != _embeddings_key:
# #         _embeddings_instance = OpenAIEmbeddings(
# #             model="text-embedding-3-large",
# #             api_key=api_key,
# #         )
# #         _embeddings_key = api_key
# #     return _embeddings_instance


# # # ── BM25 tokenizer ────────────────────────────────────────────────────────────
# # def tokenize(text: str) -> list:
# #     tokens = []
# #     for token in text.lower().split():
# #         clean = "".join(ch for ch in token if ch.isalnum() or ch in "+#")
# #         if len(clean) > 1:
# #             tokens.append(clean)
# #     return tokens


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # API CALL #1 — Parse JD with LLM
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def extract_jd_features(jd_text: str, api_key: str) -> dict:
# #     """
# #     ONE LLM call — extracts required_skills, preferred_skills, all_skills,
# #     exp_years, role_summary from ANY job description (tech or non-tech).
# #     """
# #     try:
# #         llm = ChatOpenAI(
# #             model="gpt-4o-mini",
# #             temperature=0,
# #             api_key=api_key,
# #             max_tokens=1000,
# #         )
# #         prompt = ChatPromptTemplate.from_messages([
# #             ("system",
# #              "You are a senior recruiter with expertise across all industries. "
# #              "Parse job descriptions with surgical precision. "
# #              "Output ONLY valid JSON — no markdown fences, no extra text."),
# #             ("human",
# #              "Analyze this job description and extract structured requirements.\n\n"
# #              "JD:\n{jd_text}\n\n"
# #              "Return ONLY this JSON:\n"
# #              "{{\n"
# #              "  \"required_skills\":  [\"skill\", ...],\n"
# #              "  \"preferred_skills\": [\"skill\", ...],\n"
# #              "  \"all_skills\":       [\"skill\", ...],\n"
# #              "  \"exp_years\":        <float>,\n"
# #              "  \"role_summary\":     \"<1-2 sentences>\",\n"
# #              "  \"domain\":           \"<tech|non-tech|hr|finance|sales|marketing|ops|healthcare|legal|other>\"\n"
# #              "}}\n\n"
# #              "CRITICAL RULES:\n"
# #              "• Skills must be CANONICAL CONCEPTS — the base concept, not the JD's phrasing.\n"
# #              "  Example: 'proficiency in SQL databases' → 'sql'\n"
# #              "  Example: 'experience with bug tracking tools' → 'bug tracking'\n"
# #              "  Example: 'must collaborate with cross-functional teams' → 'collaboration'\n"
# #              "  Example: 'data-driven decision making' → 'data analysis'\n"
# #              "  Example: 'managed P&L responsibilities' → 'financial management'\n"
# #              "• required_skills  — must-have / required / essential\n"
# #              "• preferred_skills — nice-to-have / preferred / bonus\n"
# #              "• all_skills       — union of everything (hard + soft + domain skills)\n"
# #              "• exp_years — minimum years required (0.0 if not mentioned)\n"
# #              "• All skill strings: lowercase, 1–4 words, canonical\n"
# #              "• Works for ANY domain — tech, sales, HR, marketing, ops, finance, legal"),
# #         ])
# #         chain  = prompt | llm | JsonOutputParser()
# #         result = chain.invoke({"jd_text": jd_text[:3000]})

# #         def clean(lst):
# #             return list(dict.fromkeys(
# #                 s.strip().lower() for s in (lst or []) if s and s.strip()
# #             ))
# #         return {
# #             "required_skills":  clean(result.get("required_skills",  [])),
# #             "preferred_skills": clean(result.get("preferred_skills", [])),
# #             "all_skills":       clean(result.get("all_skills",       [])),
# #             "exp_years":        float(result.get("exp_years", 0.0)),
# #             "role_summary":     result.get("role_summary", ""),
# #             "domain":           result.get("domain", "other"),
# #             "raw":              jd_text,
# #         }
# #     except Exception as exc:
# #         print(f"[retrieval_engine] JD parse failed: {exc}")
# #         print(traceback.format_exc())
# #         return {
# #             "required_skills": [], "preferred_skills": [], "all_skills": [],
# #             "exp_years": 0.0, "role_summary": "", "domain": "other", "raw": jd_text,
# #         }


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # INDEX — API CALL #2 happens here
# # # ═══════════════════════════════════════════════════════════════════════════════
# # LABEL_WEIGHT = {
# #     "skills": 1.4, "projects": 1.3, "experience": 1.2,
# #     "summary": 1.0, "education": 0.8, "certifications": 0.8, "full": 0.9,
# # }

# # class ResumeIndex:
# #     def __init__(self):
# #         self.candidates:    list = []
# #         self.chunk_meta:    list = []
# #         self.faiss_store:   Optional[FAISS] = None
# #         self.bm25:          Optional[BM25Okapi] = None
# #         self.corpus_tokens: list = []

# #     def build(self, parsed_resumes: list, api_key: str):
# #         """One batch call to text-embedding-3-large for ALL resume chunks."""
# #         self.candidates  = parsed_resumes
# #         self.chunk_meta  = []
# #         lc_docs          = []

# #         for idx, resume in enumerate(parsed_resumes):
# #             for label, text in resume["chunks"].items():
# #                 self.chunk_meta.append({"candidate_idx": idx, "chunk_label": label})
# #                 lc_docs.append(Document(
# #                     page_content=text,
# #                     metadata={
# #                         "candidate_idx": idx,
# #                         "chunk_label":   label,
# #                         "weight":        LABEL_WEIGHT.get(label, 1.0),
# #                     },
# #                 ))

# #         embeddings       = get_embeddings(api_key)
# #         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)

# #         self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
# #         self.bm25          = BM25Okapi(self.corpus_tokens)

# #     def is_ready(self) -> bool:
# #         return bool(self.candidates) and self.faiss_store is not None


# # # ── RRF ───────────────────────────────────────────────────────────────────────
# # def rrf(rank_lists: list, k: int = 60) -> list:
# #     scores: dict = {}
# #     for ranks in rank_lists:
# #         for rank, idx in enumerate(ranks):
# #             scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
# #     return sorted(scores.items(), key=lambda x: -x[1])


# # # ── Dense retrieval ────────────────────────────────────────────────────────────
# # def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
# #     search_k = min(len(index.chunk_meta), k * 8)
# #     hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

# #     best: dict = {}
# #     for doc, l2_dist in hits:
# #         meta     = doc.metadata
# #         cand_idx = meta["candidate_idx"]
# #         sim      = max(0.0, 1.0 - l2_dist / 2.0)
# #         wscore   = sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0)
# #         if wscore > best.get(cand_idx, -1.0):
# #             best[cand_idx] = wscore

# #     return [c for c, _ in sorted(best.items(), key=lambda x: -x[1])]


# # def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
# #     scores = index.bm25.get_scores(jd_tokens)
# #     ranked = np.argsort(-scores)
# #     return [int(i) for i in ranked[:k]]


# # # ── Heuristic pre-scoring — POOL FILTERING ONLY ───────────────────────────────
# # def heuristic_experience_score(candidate_years: float, required_years: float) -> float:
# #     if required_years <= 0:
# #         return 0.5 if candidate_years > 0 else 0.2
# #     if candidate_years <= 0:
# #         return 0.0
# #     ratio = candidate_years / required_years
# #     if ratio >= 1.0:
# #         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
# #     return max(0.0, ratio * 0.85)


# # def heuristic_skill_score(candidate_skills: list, jd_all_skills: list) -> float:
# #     """
# #     Intentionally lenient — purpose is to keep good candidates in the pool,
# #     not to rank them. LLM does precise ranking in Call #3.
# #     """
# #     if not jd_all_skills:
# #         return 0.5
# #     cand_text = " ".join(candidate_skills).lower()
# #     matches = sum(
# #         1 for skill in jd_all_skills
# #         if any(word in cand_text for word in skill.lower().split() if len(word) > 2)
# #     )
# #     return min(matches / len(jd_all_skills), 1.0)


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # API CALL #3 — LLM as authoritative skill matcher, scorer, and explainer
# # #
# # # KEY FIXES:
# # #   1. max_tokens: was 200+200*N (too low) → now min(4000, 500+400*N)
# # #   2. resume_context per candidate: was 1200 chars → now 600 chars
# # #      (enough for skill matching, stays within token budget)
# # #   3. rerank_pool: was top_n*2 (up to 20) → now min(top_n*2, 8)
# # #      (more candidates = more tokens = JSON truncated = parse fail)
# # #   4. Partial fallback: if LLM returns wrong count, apply what we have
# # #      and only fallback for remaining candidates (not all of them)
# # #   5. Full traceback logged on exception for easier debugging
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def rerank_and_explain(
# #     jd_text:         str,
# #     jd_features:     dict,
# #     candidates_pool: list,
# #     top_n:           int,
# #     api_key:         str,
# # ) -> list:
# #     """
# #     Single LLM call — does three things at once for ALL candidates:
# #       1. Semantically resolves which JD skills are present in each resume
# #       2. Assigns precise fit score (0.0–1.0)
# #       3. Generates grounded strengths + gaps
# #     """
# #     if not candidates_pool or not api_key:
# #         return candidates_pool[:top_n]

# #     # ── Build resume context — 600 chars per candidate (fits token budget) ────
# #     summaries = []
# #     for i, item in enumerate(candidates_pool):
# #         c      = item["candidate"]
# #         exp    = c.get("experience_years", 0)
# #         name   = c.get("name", f"Candidate {i}")
# #         chunks = c.get("chunks", {})

# #         # Priority order: skills → experience → summary → projects → certs
# #         resume_context = "\n".join(filter(None, [
# #             chunks.get("summary",        "")[:150],
# #             chunks.get("skills",         "")[:200],
# #             chunks.get("experience",     "")[:200],
# #             chunks.get("projects",       "")[:100],
# #             chunks.get("certifications", "")[:100],
# #         ]))[:600]  # FIX: was 1200 — 600 is enough for skill matching

# #         summaries.append(
# #             f"[{i}] {name} | {exp} yrs exp\n"
# #             f"RESUME:\n{resume_context}"
# #         )

# #     all_jd_skills = jd_features.get("all_skills", [])
# #     required      = jd_features.get("required_skills", [])

# #     n_candidates = len(candidates_pool)

# #     # FIX: max_tokens was 200+200*N — way too low, caused JSON truncation
# #     # 500 base + 400 per candidate, hard cap at 4000
# #     max_tok = min(4000, 500 + 400 * n_candidates)

# #     try:
# #         llm = ChatOpenAI(
# #             model="gpt-4o-mini",
# #             temperature=0,
# #             api_key=api_key,
# #             max_tokens=max_tok,
# #         )

# #         prompt = ChatPromptTemplate.from_messages([
# #             ("system",
# #              "You are the world's most precise recruiter across all industries. "
# #              "Your skill matching is semantically exact — you understand that resumes "
# #              "express the same skills in countless different ways depending on domain, "
# #              "seniority, company culture, and geography.\n\n"
# #              "CORE MATCHING PRINCIPLE:\n"
# #              "When checking if a JD skill is present in a resume, look for ANY of:\n"
# #              "  • The exact term\n"
# #              "  • A more specific version (Oracle SQL covers 'sql')\n"
# #              "  • A tool/platform that implies the skill (JIRA defects covers 'bug tracking')\n"
# #              "  • Demonstrated experience that proves the skill\n"
# #              "  • Industry-standard synonym or abbreviation\n"
# #              "  • Implied skill from seniority/role (QA Lead with 6+ years implies 'test planning')\n\n"
# #              "A skill is MISSING only if no form of it appears ANYWHERE in the resume.\n"
# #              "NEVER mark a skill missing if it is present in any equivalent form.\n"
# #              "NEVER invent strengths not supported by the resume text.\n"
# #              "Output ONLY valid JSON — no markdown, no extra text."),
# #             ("human",
# #              "ROLE: {role_summary}\n"
# #              "DOMAIN: {domain}\n"
# #              "JD: {jd_text}\n\n"
# #              "JD SKILLS TO MATCH (canonical concepts): {all_skills}\n"
# #              "REQUIRED SKILLS: {required_skills}\n\n"
# #              "CANDIDATES:\n{summaries}\n\n"
# #              "RULES:\n"
# #              "• Score reflects true fit: skill coverage + relevant experience + seniority match\n"
# #              "• Penalize keyword stuffing (skills listed without supporting evidence)\n"
# #              "• strengths/gaps must reference ONLY content visible in the resume above\n"
# #              "• matched_skills + missing_skills must together equal all_skills for each candidate\n"
# #              "• YOU MUST return EXACTLY {n_candidates} result objects — one per candidate\n\n"
# #              "Return ONLY valid JSON in this exact format:\n"
# #              "{{\"results\": [\n"
# #              "  {{\"score\": 0.9, "
# #              "\"matched_skills\": [...], "
# #              "\"missing_skills\": [...], "
# #              "\"required_matched\": [...], "
# #              "\"required_missing\": [...], "
# #              "\"strengths\": [\"...\", \"...\"], "
# #              "\"gaps\": [\"...\", \"...\"]}},\n"
# #              "  ...\n"
# #              "]}}"),
# #         ])

# #         chain  = prompt | llm | JsonOutputParser()
# #         output = chain.invoke({
# #             "role_summary":    jd_features.get("role_summary", ""),
# #             "domain":          jd_features.get("domain", "other"),
# #             "jd_text":         jd_text[:1500],
# #             "all_skills":      ", ".join(all_jd_skills),
# #             "required_skills": ", ".join(required),
# #             "summaries":       "\n\n---\n\n".join(summaries),
# #             "n_candidates":    n_candidates,
# #         })

# #         llm_results = output.get("results", [])

# #         # FIX: Partial fallback — apply what LLM returned, fallback only the rest
# #         # Old code: if len mismatch → raise → fallback ALL candidates
# #         # New code: apply first min(len(llm_results), n_candidates) candidates
# #         applied = 0
# #         for i, item in enumerate(candidates_pool):
# #             if i >= len(llm_results):
# #                 # Partial fallback for candidates LLM didn't return
# #                 print(f"[retrieval_engine] Partial fallback for candidate {i} — LLM returned {len(llm_results)} of {n_candidates}")
# #                 _apply_fallback(item)
# #                 continue

# #             res   = llm_results[i]
# #             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
# #             orig  = item["pre_score"]

# #             if orig < 0.10 and llm_s < 0.3:
# #                 blend = orig * llm_s
# #             else:
# #                 blend = 0.70 * llm_s + 0.30 * orig

# #             item["ce_score"]    = llm_s
# #             item["final_score"] = round(min(blend, 1.0), 4)
# #             item["explanation"] = {
# #                 "matched_skills":   res.get("matched_skills",   []),
# #                 "missing_skills":   res.get("missing_skills",   [])[:8],
# #                 "required_matched": res.get("required_matched", []),
# #                 "required_missing": res.get("required_missing", []),
# #                 "score_pct":        round(min(blend, 1.0) * 100, 1),
# #                 "strengths":        res.get("strengths", [])[:4],
# #                 "gaps":             res.get("gaps",      [])[:3],
# #             }
# #             applied += 1

# #         print(f"[retrieval_engine] LLM applied to {applied}/{n_candidates} candidates successfully")

# #     except Exception as exc:
# #         print(f"[retrieval_engine] Rerank+explain failed: {exc}")
# #         print(traceback.format_exc())   # FIX: full traceback for debugging
# #         for item in candidates_pool:
# #             _apply_fallback(item)

# #     candidates_pool.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))
# #     return candidates_pool[:top_n]


# # def _apply_fallback(item: dict):
# #     """Apply graceful fallback scoring when LLM result is unavailable."""
# #     c  = item["candidate"]
# #     ex = item["explanation"]
# #     # Keep pre_score as final_score (already set)
# #     item["ce_score"] = 0.0
# #     ex["score_pct"]  = round(item["pre_score"] * 100, 1)
# #     if not ex.get("strengths"):
# #         strengths = []
# #         if c.get("experience_years", 0) > 0:
# #             strengths.append(f"{c['experience_years']} years of relevant experience")
# #         strengths.append("Profile semantically aligns with role requirements")
# #         ex["strengths"] = strengths
# #     if not ex.get("gaps") and ex.get("required_missing"):
# #         ex["gaps"] = [f"Missing: {', '.join(ex['required_missing'][:3])}"]


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # Main entry point
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def retrieve_top_n(
# #     index:   ResumeIndex,
# #     jd_text: str,
# #     top_n:   int,
# #     api_key: str,
# # ) -> list:
# #     """
# #     Full pipeline — exactly 3 API calls total:
# #       Call #1  extract_jd_features   (LLM — JD parsing, skill canonicalization)
# #       Call #2  already in build()    (text-embedding-3-large + 1 cheap query embed)
# #       Call #3  rerank_and_explain    (LLM — semantic skill matching + scoring)
# #     """
# #     if not index.is_ready():
# #         return []

# #     # ── Call #1 ───────────────────────────────────────────────────────────────
# #     jd_features = extract_jd_features(jd_text, api_key)
# #     jd_tokens   = tokenize(jd_text)

# #     # ── Retrieval (no new LLM call) ───────────────────────────────────────────
# #     POOL_SIZE  = max(top_n * 3, 30)
# #     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
# #     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)

# #     # RRF fusion — returns (cand_idx, rrf_score) sorted by score desc
# #     fused      = rrf([sem_ranks, bm25_ranks])
# #     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

# #     # FIX: Use RRF score directly as sem_score — accurate for all candidates
# #     # Old: sem_rank_map lookup → BM25-only candidates got rank=n_cands → score=0
# #     # New: normalize RRF score (already 0-1 range) per candidate
# #     rrf_score_map = {idx: score for idx, score in fused}
# #     max_rrf = max(rrf_score_map.values()) if rrf_score_map else 1.0

# #     n_cands = max(len(index.candidates), 1)
# #     results  = []

# #     for cand_idx in cand_pool:
# #         candidate = index.candidates[cand_idx]
# #         skill_sc  = heuristic_skill_score(
# #             candidate.get("skills", []),
# #             jd_features.get("all_skills", [])
# #         )
# #         exp_sc    = heuristic_experience_score(
# #             candidate.get("experience_years", 0),
# #             jd_features["exp_years"]
# #         )

# #         # FIX: sem_score from normalized RRF score (not rank-based division)
# #         raw_rrf   = rrf_score_map.get(cand_idx, 0.0)
# #         sem_score = raw_rrf / max_rrf if max_rrf > 0 else 0.0

# #         pre_score = min(0.45 * skill_sc + 0.30 * sem_score + 0.25 * exp_sc, 1.0)

# #         results.append({
# #             "candidate":        candidate,
# #             "pre_score":        pre_score,
# #             "final_score":      pre_score,       # overwritten by LLM in Call #3
# #             "skill_score":      skill_sc,
# #             "semantic_score":   sem_score,
# #             "experience_score": exp_sc,
# #             "ce_score":         0.0,
# #             "explanation": {                     # overwritten entirely by LLM in Call #3
# #                 "matched_skills":   [],
# #                 "missing_skills":   [],
# #                 "required_matched": [],
# #                 "required_missing": [],
# #                 "score_pct":        round(pre_score * 100, 1),
# #                 "strengths":        [],
# #                 "gaps":             [],
# #             },
# #             "jd_features": jd_features,
# #         })

# #     results.sort(key=lambda x: -x["pre_score"])

# #     # FIX: Hard cap at 8 candidates for LLM call — prevents token budget bust
# #     # Old: top_n * 2 could be 20 candidates → 20 * 600 chars = 12k chars → truncated JSON
# #     rerank_pool = results[: min(top_n * 2, 8)]

# #     # ── Call #3 — LLM is the final arbiter of skill matching and scoring ──────
# #     return rerank_and_explain(jd_text, jd_features, rerank_pool, top_n, api_key)



# # this is final v3 is final
# """
# Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# ═══════════════════════════════════════════════════════════════════════════════
# FIXES (v3):
#   FIX-1  DETERMINISTIC RANKING — scores rounded to 2dp everywhere + filename
#          tiebreaker at every sort. Same JD + same resumes = same order always.
#   FIX-2  HARD SCORE GATE — if skill_score < 0.05 (zero real matches) the
#          final score is hard-capped at 0.10, LLM reranker is SKIPPED entirely,
#          and "Why Hired" shows "No relevant skills found" instead of hallucinated reasons.
#   FIX-3  JD VALIDATION — validate_jd() runs before any API call. Gibberish /
#          greetings / <10 meaningful words → returns error string, 0 API calls.
# """

# import numpy as np
# from typing import Optional
# from rank_bm25 import BM25Okapi

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser


# # ── Singleton embeddings ──────────────────────────────────────────────────────
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
# # FIX-3 : JD Validation — runs BEFORE any API call
# # ═══════════════════════════════════════════════════════════════════════════════
# _JD_STOP_WORDS = {
#     "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
#     "have", "has", "had", "do", "does", "did", "will", "would", "could",
#     "should", "may", "might", "shall", "can", "need", "dare", "ought",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
#     "their", "this", "that", "these", "those", "what", "which", "who",
#     "and", "or", "but", "nor", "so", "yet", "for", "of", "in", "on", "at",
#     "to", "by", "up", "as", "if", "into", "with", "about", "than", "then",
#     "hello", "hi", "hey", "helo", "hii", "helo", "test", "testing", "ok",
#     "okay", "yes", "no", "please", "thanks", "thank", "sure", "fine",
#     "good", "great", "nice", "wow", "hmm", "um", "uh", "lol", "haha",
# }

# _GREETINGS = {
#     "hello", "hi", "hey", "helo", "hii", "test", "testing", "ok", "okay",
#     "yo", "sup", "wassup", "whats up", "what's up", "namaste", "hola",
# }

# def validate_jd(jd_text: str) -> Optional[str]:
#     """
#     Returns an error message string if JD is invalid, else None (valid).
#     Checks:
#       1. Too short (< 10 meaningful words)
#       2. Looks like a greeting / single word
#       3. No job-relevant content (all stop words)
#     Zero API calls consumed on failure.
#     """
#     stripped = jd_text.strip().lower()

#     # Single word or greeting
#     first_word = stripped.split()[0] if stripped.split() else ""
#     if first_word in _GREETINGS and len(stripped.split()) <= 4:
#         return "❌ That looks like a greeting, not a job description. Please paste a real JD."

#     # Tokenize and filter stop words
#     all_tokens = [t.strip(".,!?;:\"'()[]") for t in stripped.split()]
#     meaningful = [t for t in all_tokens if t and t not in _JD_STOP_WORDS and len(t) > 2]

#     if len(meaningful) < 8:
#         return (
#             f"❌ JD too short — only {len(meaningful)} meaningful word(s) found. "
#             "Please paste a proper job description (role, skills, requirements)."
#         )

#     return None  # valid


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #1 — Parse JD with GPT-4o-mini
# # ═══════════════════════════════════════════════════════════════════════════════
# def extract_jd_features(jd_text: str, api_key: str) -> dict:
#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=800,
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a senior recruiter. Parse job descriptions precisely. "
#              "Output ONLY valid JSON — no markdown fences, no extra text."),
#             ("human",
#              "Analyze this job description and extract structured requirements.\n\n"
#              "JD:\n{jd_text}\n\n"
#              "Return ONLY:\n"
#              "{{\n"
#              "  \"required_skills\":  [\"skill\", ...],\n"
#              "  \"preferred_skills\": [\"skill\", ...],\n"
#              "  \"all_skills\":       [\"skill\", ...],\n"
#              "  \"skill_synonyms\":   {{\"skill\": [\"alias1\", \"alias2\"], ...}},\n"
#              "  \"exp_years\":        <float>,\n"
#              "  \"role_summary\":     \"<1-2 sentences>\",\n"
#              "  \"domain\":           \"<primary domain, e.g. software engineering, sales, logistics, marketing>\"\n"
#              "}}\n\n"
#              "Rules:\n"
#              "• required_skills  — must-have / required / essential\n"
#              "• preferred_skills — nice-to-have / preferred / bonus\n"
#              "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
#              "• skill_synonyms   — common aliases only, no generic category names as synonyms\n"
#              "• exp_years        — minimum years required (0.0 if not mentioned)\n"
#              "• domain           — single lowercase string for the primary job domain\n"
#              "• All skill strings: lowercase, 1–4 words max"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()
#         result = chain.invoke({"jd_text": jd_text[:2500]})

#         def clean(lst):
#             return list(dict.fromkeys(
#                 s.strip().lower() for s in (lst or []) if s and s.strip()
#             ))
#         return {
#             "required_skills":  clean(result.get("required_skills",  [])),
#             "preferred_skills": clean(result.get("preferred_skills", [])),
#             "all_skills":       clean(result.get("all_skills",       [])),
#             "skill_synonyms":   result.get("skill_synonyms", {}),
#             "exp_years":        float(result.get("exp_years", 0.0)),
#             "role_summary":     result.get("role_summary", ""),
#             "domain":           result.get("domain", "").lower().strip(),
#             "raw":              jd_text,
#         }
#     except Exception as exc:
#         print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
#         return {
#             "required_skills": [], "preferred_skills": [], "all_skills": [],
#             "exp_years": 0.0, "role_summary": "", "domain": "", "raw": jd_text,
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

#         # ── API CALL #2 ────────────────────────────────────────────────────────
#         embeddings       = get_embeddings(api_key)
#         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)
#         # ──────────────────────────────────────────────────────────────────────

#         self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
#         self.bm25          = BM25Okapi(self.corpus_tokens)

#     def is_ready(self) -> bool:
#         return bool(self.candidates) and self.faiss_store is not None


# # ── FIX-1 helper: stable round ────────────────────────────────────────────────
# def _r2(val: float) -> float:
#     """Round to 2 decimal places for stable sorting."""
#     return round(float(val), 2)


# # ── RRF — FIX-1: stable sort by (rrf_score desc, cand_idx asc) ───────────────
# def rrf(rank_lists: list, k: int = 60) -> list:
#     scores: dict = {}
#     for ranks in rank_lists:
#         for rank, idx in enumerate(ranks):
#             scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
#     # FIX-1: tiebreak by candidate index (stable, deterministic)
#     return sorted(scores.items(), key=lambda x: (-_r2(x[1]), x[0]))


# # ── Dense retrieval — FIX-1: round scores before storing in best{} ───────────
# def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
#     search_k = min(len(index.chunk_meta), k * 8)
#     hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

#     best: dict = {}
#     for doc, l2_dist in hits:
#         meta     = doc.metadata
#         cand_idx = meta["candidate_idx"]
#         sim      = max(0.0, 1.0 - l2_dist / 2.0)
#         wscore   = _r2(sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0))
#         if wscore > best.get(cand_idx, -1.0):
#             best[cand_idx] = wscore

#     # FIX-1: tiebreak by cand_idx
#     return [c for c, _ in sorted(best.items(), key=lambda x: (-x[1], x[0]))]


# def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
#     scores = index.bm25.get_scores(jd_tokens)
#     ranked = np.argsort(-scores)
#     return [int(i) for i in ranked[:k]]


# # ── Strict fuzzy match ────────────────────────────────────────────────────────
# def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
#     if jd_skill in cand_set:
#         return True

#     jd_words = set(jd_skill.split())

#     for cs in cand_set:
#         cs_words = set(cs.split())
#         if len(jd_words) == 1:
#             jd_word = jd_skill
#             if jd_word in cs_words:
#                 return True
#             if cs.startswith(jd_word + " ") or cs.endswith(" " + jd_word):
#                 return True
#         elif len(cs_words) == 1:
#             if cs in jd_words:
#                 return True
#         else:
#             if jd_words == cs_words:
#                 return True
#             if len(jd_words) >= 2 and jd_words.issubset(cs_words):
#                 return True
#             if len(cs_words) >= 2 and cs_words.issubset(jd_words):
#                 return True

#     if skill_synonyms:
#         for syn in skill_synonyms.get(jd_skill, []):
#             syn       = syn.lower().strip()
#             syn_words = set(syn.split())
#             if syn in cand_set:
#                 return True
#             for cs in cand_set:
#                 cs_words = set(cs.split())
#                 if len(syn_words) == 1:
#                     if syn in cs_words:
#                         return True
#                 else:
#                     if syn_words == cs_words or syn_words.issubset(cs_words):
#                         return True
#     return False


# # ── Skill overlap — minimum evidence threshold ────────────────────────────────
# def skill_overlap_score(candidate_skills: list, jd_features: dict) -> float:
#     MIN_MATCH = 2
#     all_jd    = set(jd_features.get("all_skills", []))
#     if not all_jd:
#         return 0.0

#     cand_set        = set(candidate_skills)
#     req             = set(jd_features.get("required_skills",  []))
#     pref            = set(jd_features.get("preferred_skills", []))
#     syns            = jd_features.get("skill_synonyms", {})

#     all_match_list  = [s for s in all_jd if fuzzy_match(s, cand_set, syns)]
#     req_match_list  = [s for s in req    if fuzzy_match(s, cand_set, syns)]
#     pref_match_list = [s for s in pref   if fuzzy_match(s, cand_set, syns)]

#     if len(all_match_list) < MIN_MATCH:
#         return 0.08 if req_match_list else 0.0

#     req_ratio  = len(req_match_list)  / len(req)  if req  else 0.0
#     pref_ratio = len(pref_match_list) / len(pref) if pref else 0.0
#     all_ratio  = len(all_match_list)  / len(all_jd)

#     return 0.5 * req_ratio + 0.3 * all_ratio + 0.2 * pref_ratio


# def experience_score(candidate_years: float, required_years: float) -> float:
#     if required_years <= 0:
#         return 0.5 if candidate_years > 0 else 0.2
#     if candidate_years <= 0:
#         return 0.0
#     ratio = candidate_years / required_years
#     if ratio >= 1.0:
#         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
#     return max(0.0, ratio * 0.85)


# def build_explanation_data(candidate: dict, jd_features: dict, final_score: float) -> dict:
#     cand_set  = set(candidate.get("skills", []))
#     jd_skills = set(jd_features.get("all_skills", []))
#     req       = set(jd_features.get("required_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     matched  = sorted(s for s in jd_skills if fuzzy_match(s, cand_set, syns))
#     missing  = sorted(s for s in jd_skills if not fuzzy_match(s, cand_set, syns))
#     req_done = sorted(s for s in req if fuzzy_match(s, cand_set, syns))
#     req_miss = sorted(s for s in req if not fuzzy_match(s, cand_set, syns))

#     return {
#         "matched_skills":   matched,
#         "missing_skills":   missing[:8],
#         "required_matched": req_done,
#         "required_missing": req_miss,
#         "score_pct":        round(final_score * 100, 1),
#         "strengths":        [],
#         "gaps":             [],
#     }


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIX-2 : Hard score gate — skip LLM entirely for irrelevant candidates
# # ═══════════════════════════════════════════════════════════════════════════════
# IRRELEVANT_SKILL_THRESHOLD = 0.05   # below this → hard cap, no LLM
# IRRELEVANT_SCORE_CAP       = 0.10   # max score allowed for truly irrelevant

# def _apply_hard_gate(results: list) -> tuple[list, list]:
#     """
#     Split candidates into:
#       - eligible   : skill_score >= threshold → go to LLM reranker
#       - irrelevant : skill_score <  threshold → hard-capped, no LLM call

#     Irrelevant candidates get score capped at IRRELEVANT_SCORE_CAP and
#     receive honest "No relevant skills" messaging.
#     """
#     eligible   = []
#     irrelevant = []

#     for item in results:
#         if item.get("skill_score", 0.0) < IRRELEVANT_SKILL_THRESHOLD:
#             # Hard cap — FIX-1: round
#             capped = _r2(min(item["final_score"], IRRELEVANT_SCORE_CAP))
#             item["final_score"] = capped
#             item["ce_score"]    = 0.0
#             item["explanation"]["score_pct"] = round(capped * 100, 1)
#             item["explanation"]["strengths"] = []   # no hallucinated strengths
#             item["explanation"]["gaps"]      = [
#                 "No relevant skills matched for this role",
#                 "Domain mismatch — resume not aligned to this JD",
#             ]
#             irrelevant.append(item)
#         else:
#             eligible.append(item)

#     return eligible, irrelevant


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #3 — Rerank + Explain eligible candidates only
# # ═══════════════════════════════════════════════════════════════════════════════
# def rerank_and_explain(
#     jd_text: str,
#     jd_features: dict,
#     candidates_pool: list,
#     top_n: int,
#     api_key: str,
# ) -> list:
#     """
#     FIX-2: Only called for eligible candidates (skill_score >= threshold).
#     Irrelevant ones are already capped and won't reach here.
#     FIX-1: Final sort uses (-round(score,2), filename) for determinism.
#     """
#     if not candidates_pool or not api_key:
#         return candidates_pool[:top_n]

#     summaries = []
#     for i, item in enumerate(candidates_pool):
#         c       = item["candidate"]
#         exp     = c.get("experience_years", 0)
#         name    = c.get("name", f"Candidate {i}")
#         ex      = item["explanation"]
#         matched = ", ".join(ex.get("matched_skills", [])[:15]) or "none"
#         missing = ", ".join(ex.get("missing_skills", [])[:10]) or "none"
#         chunks  = c.get("chunks", {})
#         context = (chunks.get("summary", "") + " " + chunks.get("experience", "")[:500])[:600]
#         summaries.append(
#             f"[{i}] {name} | {exp} yrs exp\n"
#             f"    CONFIRMED_PRESENT (never list in gaps): {matched}\n"
#             f"    CONFIRMED_MISSING: {missing}\n"
#             f"    context: {context}"
#         )

#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=min(4000, 600 + 400 * len(candidates_pool)),
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are an expert recruiter. Evaluate candidates strictly based on "
#              "evidence in their resume. Never invent or assume skills not shown. "
#              "Output ONLY valid JSON — no markdown, no extra text."),
#             ("human",
#              "Role: {role_summary}\n"
#              "Domain: {domain}\n"
#              "JD (first 800 chars): {jd_text}\n"
#              "Required skills: {required_skills}\n\n"
#              "Candidates:\n{summaries}\n\n"
#              "CRITICAL RULES:\n"
#              "1. CONFIRMED_PRESENT skills ARE in the resume — never list in gaps.\n"
#              "2. CONFIRMED_MISSING skills are absent — list relevant ones as gaps.\n"
#              "3. Strengths must reference ONLY evidence in candidate context. No invention.\n"
#              "4. Domain mismatch (e.g. sales resume for logistics role) → score ≤ 0.30.\n"
#              "5. Only count domain-relevant experience. Unrelated work doesn't count.\n"
#              "6. Keyword stuffing with unrelated roles → score ≤ 0.25.\n\n"
#              "For EACH candidate return:\n"
#              "  score     — float 0.0–1.0 (genuine fit)\n"
#              "  strengths — 2–3 strings under 12 words, grounded in context\n"
#              "  gaps      — 1–2 strings under 12 words, specific missing items\n\n"
#              "Return ONLY:\n"
#              "{{\"results\": [\n"
#              "  {{\"score\": 0.92, \"strengths\": [\"...\"], \"gaps\": [\"...\"]}},\n"
#              "  ...\n"
#              "]}}"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()

#         # ── API CALL #3 ────────────────────────────────────────────────────────
#         output = chain.invoke({
#             "role_summary":    jd_features.get("role_summary", ""),
#             "domain":          jd_features.get("domain", "unspecified"),
#             "jd_text":         jd_text[:800],
#             "required_skills": ", ".join(jd_features.get("required_skills", [])[:12]),
#             "summaries":       "\n".join(summaries),
#         })
#         # ──────────────────────────────────────────────────────────────────────

#         llm_results = output.get("results", [])

#         for i, item in enumerate(candidates_pool):
#             if i >= len(llm_results):
#                 break
#             res   = llm_results[i]
#             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
#             orig  = item["final_score"]

#             # Constrained blending — LLM lift is bounded by base score
#             if orig < 0.15 and llm_s < 0.4:
#                 blend = _r2(orig * llm_s)
#             else:
#                 if orig < 0.30:
#                     max_lift = 0.25
#                 elif orig < 0.55:
#                     max_lift = 0.35
#                 else:
#                     max_lift = 0.50
#                 raw_blend = 0.55 * llm_s + 0.45 * orig
#                 ceiling   = min(orig + max_lift, 1.0)
#                 blend     = _r2(min(raw_blend, ceiling))

#             item["ce_score"]    = llm_s
#             item["final_score"] = blend
#             item["explanation"]["score_pct"]  = round(blend * 100, 1)
#             item["explanation"]["strengths"]  = res.get("strengths", [])[:4]
#             item["explanation"]["gaps"]       = res.get("gaps",      [])[:3]

#     except Exception as exc:
#         print(f"[retrieval_engine] Rerank+explain failed: {exc} — using pre-ranked scores")
#         for item in candidates_pool:
#             ex = item["explanation"]
#             c  = item["candidate"]
#             if not ex["strengths"]:
#                 if ex["required_matched"]:
#                     ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
#                 if c.get("experience_years", 0) > 0:
#                     ex["strengths"].append(f"{c['experience_years']} years of experience")
#                 if ex["matched_skills"]:
#                     ex["strengths"].append(f"{len(ex['matched_skills'])} JD skills matched")
#             if not ex["gaps"] and ex["required_missing"]:
#                 ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

#     # FIX-1: deterministic sort — round to 2dp + filename tiebreaker
#     candidates_pool.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))
#     return candidates_pool[:top_n]


# # ── Domain penalty (soft, pre-LLM) ───────────────────────────────────────────
# def _apply_domain_penalty(results: list) -> list:
#     SKILL_THRESH  = 0.10
#     SEM_THRESH    = 0.25
#     PENALTY_CAP   = 0.35

#     for item in results:
#         if (item.get("skill_score", 0.0) < SKILL_THRESH and
#                 item.get("semantic_score", 0.0) < SEM_THRESH):
#             item["final_score"] = _r2(min(item["final_score"], PENALTY_CAP))
#             item["explanation"]["score_pct"] = round(item["final_score"] * 100, 1)

#     return results


# # ═══════════════════════════════════════════════════════════════════════════════
# # Main entry point
# # ═══════════════════════════════════════════════════════════════════════════════
# def retrieve_top_n(
#     index:    "ResumeIndex",
#     jd_text:  str,
#     top_n:    int,
#     api_key:  str,
# ) -> list:
#     """
#     FIX-3: validate_jd() must be called by the caller (app.py) BEFORE this
#     function — this function assumes JD is already validated.
#     """
#     if not index.is_ready():
#         return []

#     # ── Call #1 ───────────────────────────────────────────────────────────────
#     jd_features = extract_jd_features(jd_text, api_key)
#     jd_tokens   = tokenize(jd_text)

#     # ── Retrieval ─────────────────────────────────────────────────────────────
#     POOL_SIZE  = max(top_n * 3, 30)
#     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
#     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)
#     fused      = rrf([sem_ranks, bm25_ranks])
#     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

#     # ── Weighted scoring ──────────────────────────────────────────────────────
#     sem_rank_map = {cand: rank for rank, cand in enumerate(sem_ranks)}
#     n_cands      = max(len(index.candidates), 1)
#     results      = []

#     for cand_idx in cand_pool:
#         candidate = index.candidates[cand_idx]
#         skill_sc  = skill_overlap_score(candidate.get("skills", []), jd_features)
#         exp_sc    = experience_score(candidate.get("experience_years", 0), jd_features["exp_years"])

#         rank      = sem_rank_map.get(cand_idx, n_cands)
#         sem_score = max(0.0, 1.0 - rank / n_cands)
#         sem_adj   = sem_score if rank < n_cands * 0.8 else sem_score * 0.5

#         final_score = _r2(min(
#             0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0),
#             1.0,
#         ))
#         explanation = build_explanation_data(candidate, jd_features, final_score)
#         results.append({
#             "candidate":        candidate,
#             "final_score":      final_score,
#             "skill_score":      skill_sc,
#             "semantic_score":   sem_score,
#             "experience_score": exp_sc,
#             "ce_score":         0.0,
#             "explanation":      explanation,
#             "jd_features":      jd_features,
#         })

#     # FIX-1: sort with deterministic tiebreaker before splitting
#     results.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))

#     rerank_pool = results[: min(top_n * 2, 8)]

#     # FIX-2: split into eligible (go to LLM) vs irrelevant (hard-capped, skip LLM)
#     eligible, irrelevant = _apply_hard_gate(rerank_pool)

#     # Soft domain penalty on eligible candidates
#     eligible = _apply_domain_penalty(eligible)

#     # ── Call #3 — only for eligible candidates ────────────────────────────────
#     ranked_eligible = rerank_and_explain(jd_text, jd_features, eligible, top_n, api_key)

#     # Merge: eligible (reranked) + irrelevant (capped) → final sort
#     all_results = ranked_eligible + irrelevant
#     # FIX-1: final deterministic sort
#     all_results.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))

#     return all_results[:top_n]



# 1710 is final


















# """
# Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# ═══════════════════════════════════════════════════════════════════════════════
# FIXES (v3):
#   FIX-1  DETERMINISTIC RANKING — scores rounded to 2dp everywhere + filename
#          tiebreaker at every sort. Same JD + same resumes = same order always.
#   FIX-2  HARD SCORE GATE — if skill_score < 0.05 (zero real matches) the
#          final score is hard-capped at 0.10, LLM reranker is SKIPPED entirely,
#          and "Why Hired" shows "No relevant skills found" instead of hallucinated reasons.
#   FIX-3  JD VALIDATION — validate_jd() runs before any API call. Gibberish /
#          greetings / <10 meaningful words → returns error string, 0 API calls.
# """

# import numpy as np
# from typing import Optional
# from rank_bm25 import BM25Okapi

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser


# # ── Singleton embeddings ──────────────────────────────────────────────────────
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
# # FIX-3 : JD Validation — runs BEFORE any API call
# # ═══════════════════════════════════════════════════════════════════════════════
# _JD_STOP_WORDS = {
#     "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
#     "have", "has", "had", "do", "does", "did", "will", "would", "could",
#     "should", "may", "might", "shall", "can", "need", "dare", "ought",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
#     "their", "this", "that", "these", "those", "what", "which", "who",
#     "and", "or", "but", "nor", "so", "yet", "for", "of", "in", "on", "at",
#     "to", "by", "up", "as", "if", "into", "with", "about", "than", "then",
#     "hello", "hi", "hey", "helo", "hii", "helo", "test", "testing", "ok",
#     "okay", "yes", "no", "please", "thanks", "thank", "sure", "fine",
#     "good", "great", "nice", "wow", "hmm", "um", "uh", "lol", "haha",
# }

# _GREETINGS = {
#     "hello", "hi", "hey", "helo", "hii", "test", "testing", "ok", "okay",
#     "yo", "sup", "wassup", "whats up", "what's up", "namaste", "hola",
# }

# def validate_jd(jd_text: str) -> Optional[str]:
#     """
#     Returns an error message string if JD is invalid, else None (valid).
#     Checks:
#       1. Too short (< 10 meaningful words)
#       2. Looks like a greeting / single word
#       3. No job-relevant content (all stop words)
#     Zero API calls consumed on failure.
#     """
#     stripped = jd_text.strip().lower()

#     # Single word or greeting
#     first_word = stripped.split()[0] if stripped.split() else ""
#     if first_word in _GREETINGS and len(stripped.split()) <= 4:
#         return "❌ That looks like a greeting, not a job description. Please paste a real JD."

#     # Tokenize and filter stop words
#     all_tokens = [t.strip(".,!?;:\"'()[]") for t in stripped.split()]
#     meaningful = [t for t in all_tokens if t and t not in _JD_STOP_WORDS and len(t) > 2]

#     if len(meaningful) < 8:
#         return (
#             f"❌ JD too short — only {len(meaningful)} meaningful word(s) found. "
#             "Please paste a proper job description (role, skills, requirements)."
#         )

#     return None  # valid


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #1 — Parse JD with GPT-4o-mini
# # ═══════════════════════════════════════════════════════════════════════════════
# def extract_jd_features(jd_text: str, api_key: str) -> dict:
#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=800,
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a senior recruiter. Parse job descriptions precisely. "
#              "Output ONLY valid JSON — no markdown fences, no extra text."),
#             ("human",
#              "Analyze this job description and extract structured requirements.\n\n"
#              "JD:\n{jd_text}\n\n"
#              "Return ONLY:\n"
#              "{{\n"
#              "  \"required_skills\":  [\"skill\", ...],\n"
#              "  \"preferred_skills\": [\"skill\", ...],\n"
#              "  \"all_skills\":       [\"skill\", ...],\n"
#              "  \"skill_synonyms\":   {{\"skill\": [\"alias1\", \"alias2\"], ...}},\n"
#              "  \"exp_years\":        <float>,\n"
#              "  \"role_summary\":     \"<1-2 sentences>\",\n"
#              "  \"domain\":           \"<primary domain, e.g. software engineering, sales, logistics, marketing>\"\n"
#              "}}\n\n"
#              "Rules:\n"
#              "• required_skills  — must-have / required / essential\n"
#              "• preferred_skills — nice-to-have / preferred / bonus\n"
#              "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
#              "• skill_synonyms   — common aliases only, no generic category names as synonyms\n"
#              "• exp_years        — minimum years required (0.0 if not mentioned)\n"
#              "• domain           — single lowercase string for the primary job domain\n"
#              "• All skill strings: lowercase, 1–4 words max"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()
#         result = chain.invoke({"jd_text": jd_text[:2500]})

#         def clean(lst):
#             return list(dict.fromkeys(
#                 s.strip().lower() for s in (lst or []) if s and s.strip()
#             ))
#         return {
#             "required_skills":  clean(result.get("required_skills",  [])),
#             "preferred_skills": clean(result.get("preferred_skills", [])),
#             "all_skills":       clean(result.get("all_skills",       [])),
#             "skill_synonyms":   result.get("skill_synonyms", {}),
#             "exp_years":        float(result.get("exp_years", 0.0)),
#             "role_summary":     result.get("role_summary", ""),
#             "domain":           result.get("domain", "").lower().strip(),
#             "raw":              jd_text,
#         }
#     except Exception as exc:
#         print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
#         return {
#             "required_skills": [], "preferred_skills": [], "all_skills": [],
#             "exp_years": 0.0, "role_summary": "", "domain": "", "raw": jd_text,
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

#         # ── API CALL #2 ────────────────────────────────────────────────────────
#         embeddings       = get_embeddings(api_key)
#         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)
#         # ──────────────────────────────────────────────────────────────────────

#         self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
#         self.bm25          = BM25Okapi(self.corpus_tokens)

#     def is_ready(self) -> bool:
#         return bool(self.candidates) and self.faiss_store is not None


# # ── FIX-1 helper: stable round ────────────────────────────────────────────────
# def _r2(val: float) -> float:
#     """Round to 2 decimal places for stable sorting."""
#     return round(float(val), 2)


# # ── RRF — FIX-1: stable sort by (rrf_score desc, cand_idx asc) ───────────────
# def rrf(rank_lists: list, k: int = 60) -> list:
#     scores: dict = {}
#     for ranks in rank_lists:
#         for rank, idx in enumerate(ranks):
#             scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
#     # FIX-1: tiebreak by candidate index (stable, deterministic)
#     return sorted(scores.items(), key=lambda x: (-_r2(x[1]), x[0]))


# # ── Dense retrieval — FIX-1: round scores before storing in best{} ───────────
# def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
#     search_k = min(len(index.chunk_meta), k * 8)
#     hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

#     best: dict = {}
#     for doc, l2_dist in hits:
#         meta     = doc.metadata
#         cand_idx = meta["candidate_idx"]
#         sim      = max(0.0, 1.0 - l2_dist / 2.0)
#         wscore   = _r2(sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0))
#         if wscore > best.get(cand_idx, -1.0):
#             best[cand_idx] = wscore

#     # FIX-1: tiebreak by cand_idx
#     return [c for c, _ in sorted(best.items(), key=lambda x: (-x[1], x[0]))]


# def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
#     scores = index.bm25.get_scores(jd_tokens)
#     ranked = np.argsort(-scores)
#     return [int(i) for i in ranked[:k]]


# # ── Strict fuzzy match ────────────────────────────────────────────────────────
# def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
#     if jd_skill in cand_set:
#         return True

#     jd_words = set(jd_skill.split())

#     for cs in cand_set:
#         cs_words = set(cs.split())
#         if len(jd_words) == 1:
#             jd_word = jd_skill
#             if jd_word in cs_words:
#                 return True
#             if cs.startswith(jd_word + " ") or cs.endswith(" " + jd_word):
#                 return True
#         elif len(cs_words) == 1:
#             if cs in jd_words:
#                 return True
#         else:
#             if jd_words == cs_words:
#                 return True
#             if len(jd_words) >= 2 and jd_words.issubset(cs_words):
#                 return True
#             if len(cs_words) >= 2 and cs_words.issubset(jd_words):
#                 return True

#     if skill_synonyms:
#         for syn in skill_synonyms.get(jd_skill, []):
#             syn       = syn.lower().strip()
#             syn_words = set(syn.split())
#             if syn in cand_set:
#                 return True
#             for cs in cand_set:
#                 cs_words = set(cs.split())
#                 if len(syn_words) == 1:
#                     if syn in cs_words:
#                         return True
#                 else:
#                     if syn_words == cs_words or syn_words.issubset(cs_words):
#                         return True

#     # ── Built-in normalizations (supplement LLM synonyms) ────────────────────
#     _BUILTIN_ALIASES = {
#         # Frontend frameworks
#         "react":                {"reactjs", "react.js", "react js"},
#         "angular":              {"angularjs", "angular.js", "angular js", "angular 2+"},
#         "vue":                  {"vuejs", "vue.js", "vue js"},
#         "next.js":              {"nextjs", "next js"},
#         "nuxt.js":              {"nuxtjs", "nuxt js"},
#         # Styling
#         "css":                  {"css3", "css 3", "cascading style sheets"},
#         "sass":                 {"scss"},
#         "tailwind":             {"tailwindcss", "tailwind css"},
#         "bootstrap":            {"bootstrap 4", "bootstrap 5", "bootstrap3", "bootstrap4", "bootstrap5"},
#         # JavaScript / TypeScript
#         "javascript":           {"js", "es6", "es2015", "ecmascript", "vanilla js", "vanillajs"},
#         "typescript":           {"ts"},
#         # Backend / Runtime
#         "node":                 {"nodejs", "node.js", "node js"},
#         "express":              {"expressjs", "express.js"},
#         "spring boot":          {"springboot", "spring-boot"},
#         "spring":               {"spring framework", "springframework"},
#         "django":               {"django rest framework", "drf"},
#         "fastapi":              {"fast api"},
#         "flask":                {"flask api"},
#         "dotnet":               {".net", "dot net", "asp.net", "aspnet", "asp net"},
#         "dotnet core":          {".net core", "asp.net core"},
#         # Databases
#         "postgresql":           {"postgres", "psql"},
#         "mysql":                {"my sql"},
#         "mongodb":              {"mongo", "mongo db"},
#         "redis":                {"redis cache"},
#         "elasticsearch":        {"elastic search", "elastic"},
#         "mssql":                {"sql server", "microsoft sql server", "ms sql"},
#         "sqlite":               {"sqlite3"},
#         "sql":                  {"structured query language"},
#         # Cloud
#         "aws":                  {"amazon web services", "amazon aws"},
#         "azure":                {"microsoft azure"},
#         "gcp":                  {"google cloud", "google cloud platform"},
#         # DevOps / Infra
#         "docker":               {"dockerfile", "docker compose", "docker-compose"},
#         "kubernetes":           {"k8s", "kube"},
#         "jenkins":              {"jenkins ci", "jenkins pipeline"},
#         "github actions":       {"gh actions"},
#         "terraform":            {"tf"},
#         "ansible":              {"ansible playbook"},
#         "nginx":                {"nginx server"},
#         "linux":                {"ubuntu", "centos", "debian", "unix"},
#         # Version control
#         "git":                  {"github", "gitlab", "bitbucket", "version control"},
#         # AI / ML
#         "tensorflow":           {"tf", "tensor flow"},
#         "pytorch":              {"torch"},
#         "scikit-learn":         {"sklearn", "scikit learn"},
#         "opencv":               {"cv2", "open cv"},
#         "hugging face":         {"huggingface", "transformers"},
#         "langchain":            {"lang chain"},
#         # Languages
#         "python":               {"py", "python3", "python 3"},
#         "java":                 {"core java", "java se", "java ee", "j2ee"},
#         "c++":                  {"cpp", "c plus plus"},
#         "c#":                   {"csharp", "c sharp"},
#         "golang":               {"go lang", "go"},
#         "kotlin":               {"kotlin android"},
#         "swift":                {"swift ios"},
#         "rust":                 {"rust lang"},
#         "php":                  {"php7", "php8"},
#         "ruby":                 {"ruby on rails", "rails", "ror"},
#         # Mobile
#         "react native":         {"react-native"},
#         "flutter":              {"flutter dart"},
#         "android":              {"android studio", "android sdk"},
#         "ios":                  {"xcode", "swift ios"},
#         # Testing
#         "jest":                 {"jest testing"},
#         "selenium":             {"selenium webdriver"},
#         "junit":                {"junit4", "junit5"},
#         "pytest":               {"py test"},
#         "cypress":              {"cypress testing"},
#         # Project / Soft skills
#         "agile":                {"agile methodology", "agile development"},
#         "scrum":                {"scrum methodology", "scrum master"},
#         "jira":                 {"jira board"},
#         "communication skills": {"communication", "interpersonal skills"},
#         "rest api":             {"rest", "restful", "restful api", "rest apis", "rest web services"},
#         "graphql":              {"graph ql"},
#         "microservices":        {"micro services", "microservice architecture"},
#         "hibernate":            {"jpa", "java persistence api"},
#         "maven":                {"apache maven"},
#         "gradle":               {"gradle build"},
#     }
#     aliases = _BUILTIN_ALIASES.get(jd_skill, set())
#     for alias in aliases:
#         if alias in cand_set:
#             return True
#         for cs in cand_set:
#             if alias in set(cs.split()):
#                 return True

#     return False


# # ── Skill overlap — minimum evidence threshold ────────────────────────────────
# def skill_overlap_score(candidate_skills: list, jd_features: dict) -> float:
#     MIN_MATCH = 2
#     all_jd    = set(jd_features.get("all_skills", []))
#     if not all_jd:
#         return 0.0

#     cand_set        = set(candidate_skills)
#     req             = set(jd_features.get("required_skills",  []))
#     pref            = set(jd_features.get("preferred_skills", []))
#     syns            = jd_features.get("skill_synonyms", {})

#     all_match_list  = [s for s in all_jd if fuzzy_match(s, cand_set, syns)]
#     req_match_list  = [s for s in req    if fuzzy_match(s, cand_set, syns)]
#     pref_match_list = [s for s in pref   if fuzzy_match(s, cand_set, syns)]

#     if len(all_match_list) < MIN_MATCH:
#         return 0.08 if req_match_list else 0.0

#     req_ratio  = len(req_match_list)  / len(req)  if req  else 0.0
#     pref_ratio = len(pref_match_list) / len(pref) if pref else 0.0
#     all_ratio  = len(all_match_list)  / len(all_jd)

#     # ── Soft skill inference ──────────────────────────────────────────────────
#     _COMM_SIGNALS = {
#         "client interaction", "client collaboration", "requirement gathering",
#         "stakeholder", "cross-functional", "mentored", "mentoring",
#         "presentation", "client demos", "client communication",
#         "collaborated", "team lead", "leadership",
#     }
#     _SOFT_INFERENCES = {
#         "communication skills": _COMM_SIGNALS,
#         "communication":        _COMM_SIGNALS,
#         "leadership":           {"team lead", "led team", "mentored", "managed team"},
#         "teamwork":             {"cross-functional", "collaborated", "team player"},
#     }
#     bonus          = 0.0
#     cand_raw_lower = {s.lower() for s in candidate_skills}
#     for jd_sk in all_jd:
#         if jd_sk in _SOFT_INFERENCES and not fuzzy_match(jd_sk, cand_set, syns):
#             signals = _SOFT_INFERENCES[jd_sk]
#             if any(sig in cand_raw_lower or any(sig in cs for cs in cand_raw_lower) for sig in signals):
#                 bonus += (0.5 / len(all_jd))

#     return min(0.5 * req_ratio + 0.3 * all_ratio + 0.2 * pref_ratio + bonus, 1.0)


# def experience_score(candidate_years: float, required_years: float) -> float:
#     if required_years <= 0:
#         return 0.5 if candidate_years > 0 else 0.2
#     if candidate_years <= 0:
#         return 0.0
#     ratio = candidate_years / required_years
#     if ratio >= 1.0:
#         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
#     return max(0.0, ratio * 0.85)


# def build_explanation_data(candidate: dict, jd_features: dict, final_score: float) -> dict:
#     cand_set  = set(candidate.get("skills", []))
#     jd_skills = set(jd_features.get("all_skills", []))
#     req       = set(jd_features.get("required_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     matched  = sorted(s for s in jd_skills if fuzzy_match(s, cand_set, syns))
#     missing  = sorted(s for s in jd_skills if not fuzzy_match(s, cand_set, syns))
#     req_done = sorted(s for s in req if fuzzy_match(s, cand_set, syns))
#     req_miss = sorted(s for s in req if not fuzzy_match(s, cand_set, syns))

#     return {
#         "matched_skills":   matched,
#         "missing_skills":   missing[:8],
#         "required_matched": req_done,
#         "required_missing": req_miss,
#         "score_pct":        round(final_score * 100, 1),
#         "strengths":        [],
#         "gaps":             [],
#     }


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIX-2 : Hard score gate — skip LLM entirely for irrelevant candidates
# # ═══════════════════════════════════════════════════════════════════════════════
# IRRELEVANT_SKILL_THRESHOLD = 0.05   # below this → hard cap, no LLM
# IRRELEVANT_SCORE_CAP       = 0.10   # max score allowed for truly irrelevant

# def _apply_hard_gate(results: list) -> tuple[list, list]:
#     """
#     Split candidates into:
#       - eligible   : skill_score >= threshold → go to LLM reranker
#       - irrelevant : skill_score <  threshold → hard-capped, no LLM call

#     Irrelevant candidates get score capped at IRRELEVANT_SCORE_CAP and
#     receive honest "No relevant skills" messaging.
#     """
#     eligible   = []
#     irrelevant = []

#     for item in results:
#         if item.get("skill_score", 0.0) < IRRELEVANT_SKILL_THRESHOLD:
#             # Hard cap — FIX-1: round
#             capped = _r2(min(item["final_score"], IRRELEVANT_SCORE_CAP))
#             item["final_score"] = capped
#             item["ce_score"]    = 0.0
#             item["explanation"]["score_pct"] = round(capped * 100, 1)
#             item["explanation"]["strengths"] = []   # no hallucinated strengths
#             item["explanation"]["gaps"]      = [
#                 "No relevant skills matched for this role",
#                 "Domain mismatch — resume not aligned to this JD",
#             ]
#             irrelevant.append(item)
#         else:
#             eligible.append(item)

#     return eligible, irrelevant


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #3 — Rerank + Explain eligible candidates only
# # ═══════════════════════════════════════════════════════════════════════════════
# def rerank_and_explain(
#     jd_text: str,
#     jd_features: dict,
#     candidates_pool: list,
#     top_n: int,
#     api_key: str,
# ) -> list:
#     """
#     FIX-2: Only called for eligible candidates (skill_score >= threshold).
#     Irrelevant ones are already capped and won't reach here.
#     FIX-1: Final sort uses (-round(score,2), filename) for determinism.
#     """
#     if not candidates_pool or not api_key:
#         return candidates_pool[:top_n]

#     summaries = []
#     for i, item in enumerate(candidates_pool):
#         c       = item["candidate"]
#         exp     = c.get("experience_years", 0)
#         name    = c.get("name", f"Candidate {i}")
#         ex      = item["explanation"]
#         matched = ", ".join(ex.get("matched_skills", [])[:15]) or "none"
#         missing = ", ".join(ex.get("missing_skills", [])[:10]) or "none"
#         chunks  = c.get("chunks", {})
#         context = (chunks.get("summary", "") + " " + chunks.get("experience", "")[:500])[:600]
#         summaries.append(
#             f"[{i}] {name} | {exp} yrs exp\n"
#             f"    CONFIRMED_PRESENT (never list in gaps): {matched}\n"
#             f"    CONFIRMED_MISSING: {missing}\n"
#             f"    context: {context}"
#         )

#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=min(4000, 600 + 400 * len(candidates_pool)),
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are an expert recruiter. Evaluate candidates strictly based on "
#              "evidence in their resume. Never invent or assume skills not shown. "
#              "Output ONLY valid JSON — no markdown, no extra text."),
#             ("human",
#              "Role: {role_summary}\n"
#              "Domain: {domain}\n"
#              "JD (first 800 chars): {jd_text}\n"
#              "Required skills: {required_skills}\n\n"
#              "Candidates:\n{summaries}\n\n"
#              "CRITICAL RULES:\n"
#              "1. CONFIRMED_PRESENT skills ARE in the resume — never list in gaps.\n"
#              "2. CONFIRMED_MISSING skills are absent — list relevant ones as gaps.\n"
#              "3. Strengths must reference ONLY evidence in candidate context. No invention.\n"
#              "4. Domain mismatch (e.g. sales resume for logistics role) → score ≤ 0.30.\n"
#              "5. Only count domain-relevant experience. Unrelated work doesn't count.\n"
#              "6. Keyword stuffing with unrelated roles → score ≤ 0.25.\n\n"
#              "For EACH candidate return:\n"
#              "  score     — float 0.0–1.0 (genuine fit)\n"
#              "  strengths — 2–3 strings under 12 words, grounded in context\n"
#              "  gaps      — 1–2 strings under 12 words, specific missing items\n\n"
#              "Return ONLY:\n"
#              "{{\"results\": [\n"
#              "  {{\"score\": 0.92, \"strengths\": [\"...\"], \"gaps\": [\"...\"]}},\n"
#              "  ...\n"
#              "]}}"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()

#         # ── API CALL #3 ────────────────────────────────────────────────────────
#         output = chain.invoke({
#             "role_summary":    jd_features.get("role_summary", ""),
#             "domain":          jd_features.get("domain", "unspecified"),
#             "jd_text":         jd_text[:800],
#             "required_skills": ", ".join(jd_features.get("required_skills", [])[:12]),
#             "summaries":       "\n".join(summaries),
#         })
#         # ──────────────────────────────────────────────────────────────────────

#         llm_results = output.get("results", [])

#         for i, item in enumerate(candidates_pool):
#             if i >= len(llm_results):
#                 break
#             res   = llm_results[i]
#             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
#             orig  = item["final_score"]

#             # Constrained blending — LLM lift is bounded by base score
#             if orig < 0.15 and llm_s < 0.4:
#                 blend = _r2(orig * llm_s)
#             else:
#                 if orig < 0.30:
#                     max_lift = 0.25
#                 elif orig < 0.55:
#                     max_lift = 0.35
#                 else:
#                     max_lift = 0.50
#                 raw_blend = 0.55 * llm_s + 0.45 * orig
#                 ceiling   = min(orig + max_lift, 1.0)
#                 blend     = _r2(min(raw_blend, ceiling))

#             item["ce_score"]    = llm_s
#             item["final_score"] = blend
#             item["explanation"]["score_pct"]  = round(blend * 100, 1)
#             item["explanation"]["strengths"]  = res.get("strengths", [])[:4]
#             item["explanation"]["gaps"]       = res.get("gaps",      [])[:3]

#     except Exception as exc:
#         print(f"[retrieval_engine] Rerank+explain failed: {exc} — using pre-ranked scores")
#         for item in candidates_pool:
#             ex = item["explanation"]
#             c  = item["candidate"]
#             if not ex["strengths"]:
#                 if ex["required_matched"]:
#                     ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
#                 if c.get("experience_years", 0) > 0:
#                     ex["strengths"].append(f"{c['experience_years']} years of experience")
#                 if ex["matched_skills"]:
#                     ex["strengths"].append(f"{len(ex['matched_skills'])} JD skills matched")
#             if not ex["gaps"] and ex["required_missing"]:
#                 ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

#     # FIX-1: deterministic sort — round to 2dp + filename tiebreaker
#     candidates_pool.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))
#     return candidates_pool[:top_n]


# # ── Domain penalty (soft, pre-LLM) ───────────────────────────────────────────
# def _apply_domain_penalty(results: list) -> list:
#     SKILL_THRESH  = 0.10
#     SEM_THRESH    = 0.25
#     PENALTY_CAP   = 0.35

#     for item in results:
#         if (item.get("skill_score", 0.0) < SKILL_THRESH and
#                 item.get("semantic_score", 0.0) < SEM_THRESH):
#             item["final_score"] = _r2(min(item["final_score"], PENALTY_CAP))
#             item["explanation"]["score_pct"] = round(item["final_score"] * 100, 1)

#     return results


# # ═══════════════════════════════════════════════════════════════════════════════
# # Main entry point
# # ═══════════════════════════════════════════════════════════════════════════════
# def retrieve_top_n(
#     index:    "ResumeIndex",
#     jd_text:  str,
#     top_n:    int,
#     api_key:  str,
# ) -> list:
#     """
#     FIX-3: validate_jd() must be called by the caller (app.py) BEFORE this
#     function — this function assumes JD is already validated.
#     """
#     if not index.is_ready():
#         return []

#     # ── Call #1 ───────────────────────────────────────────────────────────────
#     jd_features = extract_jd_features(jd_text, api_key)
#     jd_tokens   = tokenize(jd_text)

#     # ── Retrieval ─────────────────────────────────────────────────────────────
#     POOL_SIZE  = max(top_n * 3, 30)
#     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
#     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)
#     fused      = rrf([sem_ranks, bm25_ranks])
#     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

#     # ── Weighted scoring ──────────────────────────────────────────────────────
#     sem_rank_map = {cand: rank for rank, cand in enumerate(sem_ranks)}
#     n_cands      = max(len(index.candidates), 1)
#     results      = []

#     for cand_idx in cand_pool:
#         candidate = index.candidates[cand_idx]
#         skill_sc  = skill_overlap_score(candidate.get("skills", []), jd_features)
#         exp_sc    = experience_score(candidate.get("experience_years", 0), jd_features["exp_years"])

#         rank      = sem_rank_map.get(cand_idx, n_cands)
#         sem_score = max(0.0, 1.0 - rank / n_cands)
#         sem_adj   = sem_score if rank < n_cands * 0.8 else sem_score * 0.5

#         final_score = _r2(min(
#             0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0),
#             1.0,
#         ))
#         explanation = build_explanation_data(candidate, jd_features, final_score)
#         results.append({
#             "candidate":        candidate,
#             "final_score":      final_score,
#             "skill_score":      skill_sc,
#             "semantic_score":   sem_score,
#             "experience_score": exp_sc,
#             "ce_score":         0.0,
#             "explanation":      explanation,
#             "jd_features":      jd_features,
#         })

#     # FIX-1: sort with deterministic tiebreaker before splitting
#     results.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))

#     rerank_pool = results[: min(top_n * 2, 8)]

#     # FIX-2: split into eligible (go to LLM) vs irrelevant (hard-capped, skip LLM)
#     eligible, irrelevant = _apply_hard_gate(rerank_pool)

#     # Soft domain penalty on eligible candidates
#     eligible = _apply_domain_penalty(eligible)

#     # ── Call #3 — only for eligible candidates ────────────────────────────────
#     ranked_eligible = rerank_and_explain(jd_text, jd_features, eligible, top_n, api_key)

#     # Merge: eligible (reranked) + irrelevant (capped) → final sort
#     all_results = ranked_eligible + irrelevant
#     # FIX-1: final deterministic sort
#     all_results.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))

#     return all_results[:top_n]








# """
# Retrieval Engine — LangChain · Exactly 3 API Calls Per Analysis Run
# ═══════════════════════════════════════════════════════════════════════════════
# FIXES (v4):
#   FIX-1  DETERMINISTIC RANKING — scores rounded to 2dp everywhere + filename
#          tiebreaker at every sort. Same JD + same resumes = same order always.
#   FIX-2  HARD SCORE GATE — if skill_score < 0.05 (zero real matches) the
#          final score is hard-capped at 0.10, LLM reranker is SKIPPED entirely,
#          and "Why Hired" shows "No relevant skills found" instead of hallucinated reasons.
#   FIX-3  JD VALIDATION — validate_jd() runs before any API call. Gibberish /
#          greetings / <10 meaningful words → returns error string, 0 API calls.
#   FIX-4  SYNONYM FAMILY SANITISER — _SKILL_FAMILIES maps related skills so
#          that if candidate has "sql", gaps about "postgresql/mysql" are blocked.
#          Sanitiser also expands matched_words via family lookup before filtering.
#   FIX-5  STRONGER BUILTIN ALIASES — sql↔postgresql/mysql/mssql fully linked.
# """

# import numpy as np
# from typing import Optional
# from rank_bm25 import BM25Okapi

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser


# # ── Singleton embeddings ──────────────────────────────────────────────────────
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
# # FIX-3 : JD Validation — runs BEFORE any API call
# # ═══════════════════════════════════════════════════════════════════════════════
# _JD_STOP_WORDS = {
#     "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
#     "have", "has", "had", "do", "does", "did", "will", "would", "could",
#     "should", "may", "might", "shall", "can", "need", "dare", "ought",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
#     "their", "this", "that", "these", "those", "what", "which", "who",
#     "and", "or", "but", "nor", "so", "yet", "for", "of", "in", "on", "at",
#     "to", "by", "up", "as", "if", "into", "with", "about", "than", "then",
#     "hello", "hi", "hey", "helo", "hii", "helo", "test", "testing", "ok",
#     "okay", "yes", "no", "please", "thanks", "thank", "sure", "fine",
#     "good", "great", "nice", "wow", "hmm", "um", "uh", "lol", "haha",
# }

# _GREETINGS = {
#     "hello", "hi", "hey", "helo", "hii", "test", "testing", "ok", "okay",
#     "yo", "sup", "wassup", "whats up", "what's up", "namaste", "hola",
# }

# def validate_jd(jd_text: str) -> Optional[str]:
#     """
#     Returns an error message string if JD is invalid, else None (valid).
#     Zero API calls consumed on failure.
#     """
#     stripped = jd_text.strip().lower()
#     first_word = stripped.split()[0] if stripped.split() else ""
#     if first_word in _GREETINGS and len(stripped.split()) <= 4:
#         return "❌ That looks like a greeting, not a job description. Please paste a real JD."

#     all_tokens = [t.strip(".,!?;:\"'()[]") for t in stripped.split()]
#     meaningful = [t for t in all_tokens if t and t not in _JD_STOP_WORDS and len(t) > 2]

#     if len(meaningful) < 8:
#         return (
#             f"❌ JD too short — only {len(meaningful)} meaningful word(s) found. "
#             "Please paste a proper job description (role, skills, requirements)."
#         )
#     return None


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIX-4 : Skill family map — for sanitiser expansion
# # If candidate has ANY skill in a family, gaps about OTHER skills in that family
# # are suppressed (they're effectively the same technology).
# # ═══════════════════════════════════════════════════════════════════════════════
# _SKILL_FAMILIES: dict[str, set] = {
#     # SQL / Relational databases — all treated as "knows SQL"
#     "sql_family": {
#         "sql", "postgresql", "postgres", "psql", "mysql", "my sql",
#         "mssql", "sql server", "microsoft sql server", "ms sql",
#         "sqlite", "sqlite3", "oracle", "oracle sql", "pl/sql", "plsql",
#         "t-sql", "tsql", "mariadb", "rdbms", "relational database",
#         "structured query language",
#     },
#     # NoSQL
#     "nosql_family": {
#         "mongodb", "mongo", "mongo db", "nosql", "no-sql",
#         "couchdb", "dynamodb", "cassandra", "firebase",
#     },
#     # Python web frameworks
#     "python_web_family": {
#         "django", "django rest framework", "drf", "flask", "flask api",
#         "fastapi", "fast api", "tornado", "aiohttp",
#     },
#     # JavaScript runtimes / backend
#     "node_family": {
#         "node", "nodejs", "node.js", "node js", "express", "expressjs",
#         "express.js", "nestjs", "nest.js",
#     },
#     # React ecosystem
#     "react_family": {
#         "react", "reactjs", "react.js", "react js",
#         "next.js", "nextjs", "next js", "remix",
#     },
#     # Vue ecosystem
#     "vue_family": {
#         "vue", "vuejs", "vue.js", "vue js", "nuxt.js", "nuxtjs",
#     },
#     # Cloud — treat any cloud exp as partial overlap
#     "cloud_family": {
#         "aws", "amazon web services", "azure", "microsoft azure",
#         "gcp", "google cloud", "google cloud platform", "cloud",
#     },
#     # Container / orchestration
#     "container_family": {
#         "docker", "kubernetes", "k8s", "kube", "docker compose",
#         "container", "containerization",
#     },
#     # Git / version control
#     "git_family": {
#         "git", "github", "gitlab", "bitbucket", "version control",
#     },
#     # Python language variants
#     "python_family": {
#         "python", "python3", "python 3", "py",
#     },
#     # Java ecosystem
#     "java_family": {
#         "java", "core java", "java se", "java ee", "j2ee",
#         "spring", "spring boot", "springboot", "spring framework",
#         "hibernate", "jpa",
#     },
#     # .NET ecosystem
#     "dotnet_family": {
#         "dotnet", ".net", "dot net", "c#", "csharp", "asp.net",
#         "aspnet", ".net core", "dotnet core",
#     },
#     # JavaScript / TypeScript
#     "js_family": {
#         "javascript", "js", "es6", "ecmascript", "typescript", "ts",
#         "vanilla js", "vanillajs",
#     },
#     # CSS / styling
#     "css_family": {
#         "css", "css3", "sass", "scss", "tailwind", "tailwindcss",
#         "bootstrap", "styled components",
#     },
#     # REST / API
#     "rest_family": {
#         "rest", "rest api", "restful", "restful api", "rest apis",
#         "rest web services", "api", "web services",
#     },
#     # ML / AI frameworks
#     "ml_family": {
#         "machine learning", "ml", "deep learning", "dl",
#         "tensorflow", "pytorch", "scikit-learn", "sklearn",
#         "keras", "xgboost", "neural network",
#     },
#     # Linux / OS
#     "linux_family": {
#         "linux", "ubuntu", "centos", "debian", "unix", "bash",
#         "shell scripting", "shell script",
#     },
# }

# # Build reverse lookup: skill_token → family_name
# _SKILL_TO_FAMILY: dict[str, str] = {}
# for _fam_name, _fam_skills in _SKILL_FAMILIES.items():
#     for _sk in _fam_skills:
#         _SKILL_TO_FAMILY[_sk] = _fam_name


# def _expand_matched_words_via_families(matched_skills: list) -> set:
#     """
#     Given a list of matched skill strings, return an expanded set of words
#     that includes all members of the same skill family.
#     E.g. if "sql" is matched, adds "postgresql", "mysql", etc. to the word set
#     so the sanitiser blocks gaps like "No PostgreSQL experience".
#     """
#     expanded: set = set()
#     for sk in matched_skills:
#         sk_lower = sk.lower().strip()
#         expanded.update(sk_lower.split())
#         # Add all family members
#         fam = _SKILL_TO_FAMILY.get(sk_lower)
#         if fam:
#             for family_skill in _SKILL_FAMILIES[fam]:
#                 expanded.update(family_skill.lower().split())
#     return expanded


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #1 — Parse JD with GPT-4o-mini
# # ═══════════════════════════════════════════════════════════════════════════════
# def extract_jd_features(jd_text: str, api_key: str) -> dict:
#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=800,
#             model_kwargs={"seed": 42},
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a senior recruiter. Parse job descriptions precisely. "
#              "Output ONLY valid JSON — no markdown fences, no extra text."),
#             ("human",
#              "Analyze this job description and extract structured requirements.\n\n"
#              "JD:\n{jd_text}\n\n"
#              "Return ONLY:\n"
#              "{{\n"
#              "  \"required_skills\":  [\"skill\", ...],\n"
#              "  \"preferred_skills\": [\"skill\", ...],\n"
#              "  \"all_skills\":       [\"skill\", ...],\n"
#              "  \"skill_synonyms\":   {{\"skill\": [\"alias1\", \"alias2\"], ...}},\n"
#              "  \"exp_years\":        <float>,\n"
#              "  \"role_summary\":     \"<1-2 sentences>\",\n"
#              "  \"domain\":           \"<primary domain, e.g. software engineering, sales, logistics, marketing>\"\n"
#              "}}\n\n"
#              "Rules:\n"
#              "• required_skills  — must-have / required / essential\n"
#              "• preferred_skills — nice-to-have / preferred / bonus\n"
#              "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
#              "• skill_synonyms   — common aliases only, no generic category names as synonyms\n"
#              "• exp_years        — minimum years required (0.0 if not mentioned)\n"
#              "• domain           — single lowercase string for the primary job domain\n"
#              "• All skill strings: lowercase, 1–4 words max"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()
#         result = chain.invoke({"jd_text": jd_text[:2500]})

#         def clean(lst):
#             return list(dict.fromkeys(
#                 s.strip().lower() for s in (lst or []) if s and s.strip()
#             ))
#         return {
#             "required_skills":  clean(result.get("required_skills",  [])),
#             "preferred_skills": clean(result.get("preferred_skills", [])),
#             "all_skills":       clean(result.get("all_skills",       [])),
#             "skill_synonyms":   result.get("skill_synonyms", {}),
#             "exp_years":        float(result.get("exp_years", 0.0)),
#             "role_summary":     result.get("role_summary", ""),
#             "domain":           result.get("domain", "").lower().strip(),
#             "raw":              jd_text,
#         }
#     except Exception as exc:
#         print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
#         return {
#             "required_skills": [], "preferred_skills": [], "all_skills": [],
#             "exp_years": 0.0, "role_summary": "", "domain": "", "raw": jd_text,
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

#         # ── API CALL #2 ────────────────────────────────────────────────────────
#         embeddings       = get_embeddings(api_key)
#         self.faiss_store = FAISS.from_documents(lc_docs, embeddings)
#         # ──────────────────────────────────────────────────────────────────────

#         self.corpus_tokens = [tokenize(r["raw_text"]) for r in parsed_resumes]
#         self.bm25          = BM25Okapi(self.corpus_tokens)

#     def is_ready(self) -> bool:
#         return bool(self.candidates) and self.faiss_store is not None


# # ── FIX-1 helper: stable round ────────────────────────────────────────────────
# def _r2(val: float) -> float:
#     return round(float(val), 2)


# # ── RRF — deterministic sort ──────────────────────────────────────────────────
# def rrf(rank_lists: list, k: int = 60) -> list:
#     scores: dict = {}
#     for ranks in rank_lists:
#         for rank, idx in enumerate(ranks):
#             scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
#     return sorted(scores.items(), key=lambda x: (-_r2(x[1]), x[0]))


# def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
#     search_k = min(len(index.chunk_meta), k * 8)
#     hits     = index.faiss_store.similarity_search_with_score(jd_text, k=search_k)

#     best: dict = {}
#     for doc, l2_dist in hits:
#         meta     = doc.metadata
#         cand_idx = meta["candidate_idx"]
#         sim      = max(0.0, 1.0 - l2_dist / 2.0)
#         wscore   = _r2(sim * LABEL_WEIGHT.get(meta.get("chunk_label", "full"), 1.0))
#         if wscore > best.get(cand_idx, -1.0):
#             best[cand_idx] = wscore

#     return [c for c, _ in sorted(best.items(), key=lambda x: (-x[1], x[0]))]


# def bm25_top_k(index: ResumeIndex, jd_tokens: list, k: int) -> list:
#     scores = index.bm25.get_scores(jd_tokens)
#     ranked = np.argsort(-scores)
#     return [int(i) for i in ranked[:k]]


# # ── Strict fuzzy match ────────────────────────────────────────────────────────
# def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
#     if jd_skill in cand_set:
#         return True

#     jd_words = set(jd_skill.split())

#     for cs in cand_set:
#         cs_words = set(cs.split())
#         if len(jd_words) == 1:
#             jd_word = jd_skill
#             if jd_word in cs_words:
#                 return True
#             if cs.startswith(jd_word + " ") or cs.endswith(" " + jd_word):
#                 return True
#         elif len(cs_words) == 1:
#             if cs in jd_words:
#                 return True
#         else:
#             if jd_words == cs_words:
#                 return True
#             if len(jd_words) >= 2 and jd_words.issubset(cs_words):
#                 return True
#             if len(cs_words) >= 2 and cs_words.issubset(jd_words):
#                 return True

#     if skill_synonyms:
#         for syn in skill_synonyms.get(jd_skill, []):
#             syn       = syn.lower().strip()
#             syn_words = set(syn.split())
#             if syn in cand_set:
#                 return True
#             for cs in cand_set:
#                 cs_words = set(cs.split())
#                 if len(syn_words) == 1:
#                     if syn in cs_words:
#                         return True
#                 else:
#                     if syn_words == cs_words or syn_words.issubset(cs_words):
#                         return True

#     # ── Built-in normalizations ───────────────────────────────────────────────
#     _BUILTIN_ALIASES = {
#         # Frontend frameworks
#         "react":                {"reactjs", "react.js", "react js"},
#         "angular":              {"angularjs", "angular.js", "angular js", "angular 2+"},
#         "vue":                  {"vuejs", "vue.js", "vue js"},
#         "next.js":              {"nextjs", "next js"},
#         "nuxt.js":              {"nuxtjs", "nuxt js"},
#         # Styling
#         "css":                  {"css3", "css 3", "cascading style sheets"},
#         "sass":                 {"scss"},
#         "tailwind":             {"tailwindcss", "tailwind css"},
#         "bootstrap":            {"bootstrap 4", "bootstrap 5", "bootstrap3", "bootstrap4", "bootstrap5"},
#         # JavaScript / TypeScript
#         "javascript":           {"js", "es6", "es2015", "ecmascript", "vanilla js", "vanillajs"},
#         "typescript":           {"ts"},
#         # Backend / Runtime
#         "node":                 {"nodejs", "node.js", "node js"},
#         "express":              {"expressjs", "express.js"},
#         "spring boot":          {"springboot", "spring-boot"},
#         "spring":               {"spring framework", "springframework"},
#         "django":               {"django rest framework", "drf"},
#         "fastapi":              {"fast api"},
#         "flask":                {"flask api"},
#         "dotnet":               {".net", "dot net", "asp.net", "aspnet", "asp net"},
#         "dotnet core":          {".net core", "asp.net core"},
#         # Databases — FIX-5: sql now aliases ALL relational DB names
#         "sql":                  {
#             "structured query language", "rdbms", "relational database",
#             "postgresql", "postgres", "psql",
#             "mysql", "my sql",
#             "mssql", "sql server", "microsoft sql server", "ms sql",
#             "sqlite", "sqlite3",
#             "oracle", "oracle sql", "pl/sql", "plsql",
#             "t-sql", "tsql", "mariadb",
#         },
#         "postgresql":           {"postgres", "psql", "sql", "rdbms"},
#         "mysql":                {"my sql", "sql", "rdbms"},
#         "mongodb":              {"mongo", "mongo db", "nosql"},
#         "redis":                {"redis cache"},
#         "elasticsearch":        {"elastic search", "elastic"},
#         "mssql":                {"sql server", "microsoft sql server", "ms sql", "sql"},
#         "sqlite":               {"sqlite3", "sql"},
#         # Cloud
#         "aws":                  {"amazon web services", "amazon aws"},
#         "azure":                {"microsoft azure"},
#         "gcp":                  {"google cloud", "google cloud platform"},
#         # DevOps / Infra
#         "docker":               {"dockerfile", "docker compose", "docker-compose"},
#         "kubernetes":           {"k8s", "kube"},
#         "jenkins":              {"jenkins ci", "jenkins pipeline"},
#         "github actions":       {"gh actions"},
#         "terraform":            {"tf"},
#         "ansible":              {"ansible playbook"},
#         "nginx":                {"nginx server"},
#         "linux":                {"ubuntu", "centos", "debian", "unix"},
#         # Version control
#         "git":                  {"github", "gitlab", "bitbucket", "version control"},
#         # AI / ML
#         "tensorflow":           {"tf", "tensor flow"},
#         "pytorch":              {"torch"},
#         "scikit-learn":         {"sklearn", "scikit learn"},
#         "opencv":               {"cv2", "open cv"},
#         "hugging face":         {"huggingface", "transformers"},
#         "langchain":            {"lang chain"},
#         # Languages
#         "python":               {"py", "python3", "python 3"},
#         "java":                 {"core java", "java se", "java ee", "j2ee"},
#         "c++":                  {"cpp", "c plus plus"},
#         "c#":                   {"csharp", "c sharp"},
#         "golang":               {"go lang", "go"},
#         "kotlin":               {"kotlin android"},
#         "swift":                {"swift ios"},
#         "rust":                 {"rust lang"},
#         "php":                  {"php7", "php8"},
#         "ruby":                 {"ruby on rails", "rails", "ror"},
#         # Mobile
#         "react native":         {"react-native"},
#         "flutter":              {"flutter dart"},
#         "android":              {"android studio", "android sdk"},
#         "ios":                  {"xcode", "swift ios"},
#         # Testing
#         "jest":                 {"jest testing"},
#         "selenium":             {"selenium webdriver"},
#         "junit":                {"junit4", "junit5"},
#         "pytest":               {"py test"},
#         "cypress":              {"cypress testing"},
#         # Project / Soft skills
#         "agile":                {"agile methodology", "agile development"},
#         "scrum":                {"scrum methodology", "scrum master"},
#         "jira":                 {"jira board"},
#         "communication skills": {"communication", "interpersonal skills"},
#         "rest api":             {"rest", "restful", "restful api", "rest apis", "rest web services"},
#         "graphql":              {"graph ql"},
#         "microservices":        {"micro services", "microservice architecture"},
#         "hibernate":            {"jpa", "java persistence api"},
#         "maven":                {"apache maven"},
#         "gradle":               {"gradle build"},
#     }
#     aliases = _BUILTIN_ALIASES.get(jd_skill, set())
#     for alias in aliases:
#         if alias in cand_set:
#             return True
#         for cs in cand_set:
#             if alias in set(cs.split()):
#                 return True

#     return False


# # ── Skill overlap — verified/claimed weighted scoring ────────────────────────
# def skill_overlap_score(candidate: dict, jd_features: dict) -> float:
#     """
#     verified_skills (proven in experience/projects) -> full credit (1.0x)
#     claimed_skills  (only in skills section)        -> half credit (0.5x)
#     """
#     MIN_MATCH = 2
#     all_jd    = set(jd_features.get("all_skills", []))
#     if not all_jd:
#         return 0.0

#     verified  = set(candidate.get("verified_skills", candidate.get("skills", [])))
#     claimed   = set(candidate.get("claimed_skills",  []))
#     req       = set(jd_features.get("required_skills",  []))
#     pref      = set(jd_features.get("preferred_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     def match_score(jd_sk: str) -> float:
#         if fuzzy_match(jd_sk, verified, syns):
#             return 1.0
#         if fuzzy_match(jd_sk, claimed, syns):
#             return 0.5
#         return 0.0

#     all_scores  = [match_score(s) for s in all_jd]
#     req_scores  = [match_score(s) for s in req]
#     pref_scores = [match_score(s) for s in pref]

#     total_matched = sum(1 for sc in all_scores if sc > 0)
#     if total_matched < MIN_MATCH:
#         return 0.08 if any(sc > 0 for sc in req_scores) else 0.0

#     all_ratio  = sum(all_scores)  / len(all_jd)
#     req_ratio  = sum(req_scores)  / len(req)  if req  else 0.0
#     pref_ratio = sum(pref_scores) / len(pref) if pref else 0.0

#     # ── Soft skill inference ──────────────────────────────────────────────────
#     _COMM_SIGNALS = {
#         "client interaction", "client collaboration", "requirement gathering",
#         "stakeholder", "cross-functional", "mentored", "mentoring",
#         "presentation", "client demos", "client communication",
#         "collaborated", "team lead", "leadership",
#     }
#     _SOFT_INFERENCES = {
#         "communication skills": _COMM_SIGNALS,
#         "communication":        _COMM_SIGNALS,
#         "leadership":           {"team lead", "led team", "mentored", "managed team"},
#         "teamwork":             {"cross-functional", "collaborated", "team player"},
#     }
#     bonus          = 0.0
#     all_cand_lower = {s.lower() for s in (verified | claimed)}
#     for jd_sk in all_jd:
#         if jd_sk in _SOFT_INFERENCES and match_score(jd_sk) == 0.0:
#             signals = _SOFT_INFERENCES[jd_sk]
#             if any(sig in all_cand_lower or any(sig in cs for cs in all_cand_lower) for sig in signals):
#                 bonus += (0.5 / len(all_jd))

#     return min(0.5 * req_ratio + 0.3 * all_ratio + 0.2 * pref_ratio + bonus, 1.0)


# def experience_score(candidate_years: float, required_years: float) -> float:
#     if required_years <= 0:
#         return 0.5 if candidate_years > 0 else 0.2
#     if candidate_years <= 0:
#         return 0.0
#     ratio = candidate_years / required_years
#     if ratio >= 1.0:
#         return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
#     return max(0.0, ratio * 0.85)


# def build_explanation_data(candidate: dict, jd_features: dict, final_score: float) -> dict:
#     verified  = set(candidate.get("verified_skills", candidate.get("skills", [])))
#     claimed   = set(candidate.get("claimed_skills",  []))
#     all_cand  = verified | claimed
#     jd_skills = set(jd_features.get("all_skills", []))
#     req       = set(jd_features.get("required_skills", []))
#     syns      = jd_features.get("skill_synonyms", {})

#     matched_verified = sorted(s for s in jd_skills if fuzzy_match(s, verified, syns))
#     matched_claimed  = sorted(s for s in jd_skills if not fuzzy_match(s, verified, syns) and fuzzy_match(s, claimed, syns))
#     # FIX: a skill is only "missing" if it cannot be matched by verified, claimed,
#     # OR any skill in the same family as what the candidate has.
#     all_matched_set  = set(matched_verified) | set(matched_claimed)
#     missing = sorted(
#         s for s in jd_skills
#         if not fuzzy_match(s, all_cand, syns) and not _family_covered(s, all_cand)
#     )
#     req_done = sorted(s for s in req if fuzzy_match(s, all_cand, syns) or _family_covered(s, all_cand))
#     req_miss = sorted(s for s in req if s not in set(req_done))

#     return {
#         "matched_skills":   matched_verified,
#         "claimed_skills":   matched_claimed,
#         "missing_skills":   missing[:8],
#         "required_matched": req_done,
#         "required_missing": req_miss,
#         "score_pct":        round(final_score * 100, 1),
#         "strengths":        [],
#         "gaps":             [],
#     }


# def _family_covered(jd_skill: str, cand_skills: set) -> bool:
#     """
#     Returns True if the candidate has any skill in the same family as jd_skill.
#     E.g. if JD asks for postgresql and candidate has sql → True (same sql_family).
#     """
#     jd_fam = _SKILL_TO_FAMILY.get(jd_skill.lower())
#     if not jd_fam:
#         return False
#     fam_skills = _SKILL_FAMILIES[jd_fam]
#     for cs in cand_skills:
#         if cs.lower() in fam_skills:
#             return True
#     return False


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIX-2 : Hard score gate
# # ═══════════════════════════════════════════════════════════════════════════════
# IRRELEVANT_SKILL_THRESHOLD = 0.05
# IRRELEVANT_SCORE_CAP       = 0.10

# def _apply_hard_gate(results: list) -> tuple[list, list]:
#     eligible   = []
#     irrelevant = []
#     for item in results:
#         if item.get("skill_score", 0.0) < IRRELEVANT_SKILL_THRESHOLD:
#             capped = _r2(min(item["final_score"], IRRELEVANT_SCORE_CAP))
#             item["final_score"] = capped
#             item["ce_score"]    = 0.0
#             item["explanation"]["score_pct"] = round(capped * 100, 1)
#             item["explanation"]["strengths"] = []
#             item["explanation"]["gaps"]      = [
#                 "No relevant skills matched for this role",
#                 "Domain mismatch — resume not aligned to this JD",
#             ]
#             irrelevant.append(item)
#         else:
#             eligible.append(item)
#     return eligible, irrelevant


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIX-4 : Hard post-processing sanitiser — removes contradictions from LLM output
# # ═══════════════════════════════════════════════════════════════════════════════
# def _sanitise_llm_output(strengths: list, gaps: list, ex: dict) -> tuple[list, list]:
#     """
#     Removes contradictions from LLM-generated strengths and gaps.

#     Rules:
#     1. A strength containing words from missing_skills → removed.
#     2. A gap containing words from matched_skills → removed.
#     3. FIX-4: Gaps about skills in the same FAMILY as a matched skill → removed.
#        (e.g. matched=sql → blocks gap about postgresql, mysql, etc.)
#     4. Gaps about required_matched skills → removed.
#     """
#     IGNORE = {
#         "in", "and", "or", "the", "a", "an", "of", "with", "for",
#         "is", "are", "has", "have", "good", "strong", "knowledge",
#         "experience", "skills", "skill", "using", "use", "used",
#         "no", "not", "missing", "lack", "lacks", "limited", "proficient",
#         "background", "exposure", "hands", "on", "hands-on",
#     }

#     # Words that indicate "candidate has this skill" — block from gaps
#     all_matched = (
#         ex.get("matched_skills", [])
#         + ex.get("claimed_skills", [])
#         + ex.get("required_matched", [])
#     )
#     # FIX-4: expand matched words to include full skill families
#     matched_expanded = _expand_matched_words_via_families(all_matched)

#     missing_words: set = set()
#     for sk in ex.get("missing_skills", []):
#         missing_words.update(sk.lower().split())

#     def _words(text: str) -> set:
#         return set(text.lower().split()) - IGNORE

#     def _has_missing_word(text: str) -> bool:
#         return bool(_words(text) & (missing_words - IGNORE))

#     def _has_matched_word(text: str) -> bool:
#         return bool(_words(text) & (matched_expanded - IGNORE))

#     clean_strengths = [s for s in strengths if not _has_missing_word(s)]
#     clean_gaps      = [g for g in gaps      if not _has_matched_word(g)]

#     return clean_strengths, clean_gaps


# # ═══════════════════════════════════════════════════════════════════════════════
# # API CALL #3 — Rerank + Explain eligible candidates only
# # ═══════════════════════════════════════════════════════════════════════════════
# def rerank_and_explain(
#     jd_text: str,
#     jd_features: dict,
#     candidates_pool: list,
#     top_n: int,
#     api_key: str,
# ) -> list:
#     if not candidates_pool or not api_key:
#         return candidates_pool[:top_n]

#     summaries = []
#     for i, item in enumerate(candidates_pool):
#         c       = item["candidate"]
#         exp     = c.get("experience_years", 0)
#         name    = c.get("name", f"Candidate {i}")
#         ex      = item["explanation"]
#         present = ex.get("matched_skills", []) + ex.get("claimed_skills", [])
#         matched = ", ".join(present[:15]) or "none"
#         missing = ", ".join(ex.get("missing_skills", [])[:10]) or "none"
#         chunks  = c.get("chunks", {})
#         context = (chunks.get("summary", "") + " " + chunks.get("experience", "")[:500])[:600]
#         summaries.append(
#             f"[{i}] {name} | {exp} yrs exp\n"
#             f"    CONFIRMED_PRESENT — skills verified in resume, NEVER put in gaps: {matched}\n"
#             f"    CONFIRMED_MISSING — skills absent from resume, ONLY use for gaps: {missing}\n"
#             f"    context: {context}"
#         )

#     try:
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             api_key=api_key,
#             max_tokens=min(4000, 600 + 400 * len(candidates_pool)),
#         )
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are an expert recruiter. Output ONLY valid JSON. No markdown, no extra text.\n"
#              "ABSOLUTE RULE: Skills in CONFIRMED_PRESENT → CANNOT appear in gaps. Ever.\n"
#              "ABSOLUTE RULE: Skills in CONFIRMED_MISSING → CANNOT appear in strengths. Ever.\n"
#              "ABSOLUTE RULE: Do NOT invent or assume skills not listed in CONFIRMED_PRESENT.\n"
#              "Violating these rules makes output useless."),
#             ("human",
#              "Role: {role_summary}\n"
#              "Domain: {domain}\n"
#              "JD: {jd_text}\n"
#              "Required: {required_skills}\n\n"
#              "Candidates:\n{summaries}\n\n"
#              "STRICT RULES:\n"
#              "1. Strengths — ONLY reference skills/experience from CONFIRMED_PRESENT. Never invent.\n"
#              "2. Gaps — ONLY reference skills from CONFIRMED_MISSING. Never from CONFIRMED_PRESENT.\n"
#              "3. Domain mismatch → score ≤ 0.30.\n"
#              "4. Keyword stuffing (many skills listed, unrelated job roles) → score ≤ 0.25.\n"
#              "5. Only count domain-relevant experience.\n\n"
#              "For EACH candidate:\n"
#              "  score     — float 0.0–1.0\n"
#              "  strengths — 2–3 strings max 12 words each, ONLY from CONFIRMED_PRESENT\n"
#              "  gaps      — 1–2 strings max 12 words each, ONLY from CONFIRMED_MISSING\n\n"
#              "Return ONLY:\n"
#              "{{\"results\": [{{\"score\": 0.9, \"strengths\": [\"...\"], \"gaps\": [\"...\"]}}, ...]}}"),
#         ])
#         chain  = prompt | llm | JsonOutputParser()

#         output = chain.invoke({
#             "role_summary":    jd_features.get("role_summary", ""),
#             "domain":          jd_features.get("domain", "unspecified"),
#             "jd_text":         jd_text[:800],
#             "required_skills": ", ".join(jd_features.get("required_skills", [])[:12]),
#             "summaries":       "\n".join(summaries),
#         })

#         llm_results = output.get("results", [])

#         for i, item in enumerate(candidates_pool):
#             if i >= len(llm_results):
#                 break
#             res   = llm_results[i]
#             llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
#             orig  = item["final_score"]

#             if orig < 0.15 and llm_s < 0.4:
#                 blend = _r2(orig * llm_s)
#             else:
#                 if orig < 0.30:
#                     max_lift = 0.25
#                 elif orig < 0.55:
#                     max_lift = 0.35
#                 else:
#                     max_lift = 0.50
#                 raw_blend = 0.55 * llm_s + 0.45 * orig
#                 ceiling   = min(orig + max_lift, 1.0)
#                 blend     = _r2(min(raw_blend, ceiling))

#             raw_strengths = res.get("strengths", [])[:4]
#             raw_gaps      = res.get("gaps",      [])[:3]

#             # ── FIX-4: Hard post-processing — remove ALL contradictions ───────
#             ex = item["explanation"]
#             clean_strengths, clean_gaps = _sanitise_llm_output(raw_strengths, raw_gaps, ex)

#             item["ce_score"]    = llm_s
#             item["final_score"] = blend
#             item["explanation"]["score_pct"] = round(blend * 100, 1)
#             item["explanation"]["strengths"] = clean_strengths
#             item["explanation"]["gaps"]      = clean_gaps

#     except Exception as exc:
#         print(f"[retrieval_engine] Rerank+explain failed: {exc} — fallback")
#         for item in candidates_pool:
#             ex = item["explanation"]
#             c  = item["candidate"]
#             if not ex["strengths"]:
#                 if ex.get("required_matched"):
#                     ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
#                 if c.get("experience_years", 0) > 0:
#                     ex["strengths"].append(f"{c['experience_years']} years of experience")
#             if not ex["gaps"] and ex.get("required_missing"):
#                 ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

#     candidates_pool.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))
#     return candidates_pool[:top_n]


# # ── Domain penalty (soft, pre-LLM) ───────────────────────────────────────────
# def _apply_domain_penalty(results: list) -> list:
#     SKILL_THRESH  = 0.10
#     SEM_THRESH    = 0.25
#     PENALTY_CAP   = 0.35

#     for item in results:
#         if (item.get("skill_score", 0.0) < SKILL_THRESH and
#                 item.get("semantic_score", 0.0) < SEM_THRESH):
#             item["final_score"] = _r2(min(item["final_score"], PENALTY_CAP))
#             item["explanation"]["score_pct"] = round(item["final_score"] * 100, 1)

#     return results


# # ═══════════════════════════════════════════════════════════════════════════════
# # Main entry point
# # ═══════════════════════════════════════════════════════════════════════════════
# def retrieve_top_n(
#     index:    "ResumeIndex",
#     jd_text:  str,
#     top_n:    int,
#     api_key:  str,
# ) -> list:
#     if not index.is_ready():
#         return []

#     jd_features = extract_jd_features(jd_text, api_key)
#     jd_tokens   = tokenize(jd_text)

#     POOL_SIZE  = max(top_n * 3, 30)
#     sem_ranks  = semantic_top_k(index, jd_text, POOL_SIZE)
#     bm25_ranks = bm25_top_k(index, jd_tokens, POOL_SIZE)
#     fused      = rrf([sem_ranks, bm25_ranks])
#     cand_pool  = [idx for idx, _ in fused[:POOL_SIZE]]

#     sem_rank_map = {cand: rank for rank, cand in enumerate(sem_ranks)}
#     n_cands      = max(len(index.candidates), 1)
#     results      = []

#     for cand_idx in cand_pool:
#         candidate = index.candidates[cand_idx]
#         skill_sc  = skill_overlap_score(candidate, jd_features)
#         exp_sc    = experience_score(candidate.get("experience_years", 0), jd_features["exp_years"])

#         rank      = sem_rank_map.get(cand_idx, n_cands)
#         sem_score = max(0.0, 1.0 - rank / n_cands)
#         sem_adj   = sem_score if rank < n_cands * 0.8 else sem_score * 0.5

#         final_score = _r2(min(
#             0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0),
#             1.0,
#         ))
#         explanation = build_explanation_data(candidate, jd_features, final_score)
#         results.append({
#             "candidate":        candidate,
#             "final_score":      final_score,
#             "skill_score":      skill_sc,
#             "semantic_score":   sem_score,
#             "experience_score": exp_sc,
#             "ce_score":         0.0,
#             "explanation":      explanation,
#             "jd_features":      jd_features,
#         })

#     results.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))

#     rerank_pool = results[: min(top_n * 2, 8)]

#     eligible, irrelevant = _apply_hard_gate(rerank_pool)
#     eligible = _apply_domain_penalty(eligible)

#     ranked_eligible = rerank_and_explain(jd_text, jd_features, eligible, top_n, api_key)

#     all_results = ranked_eligible + irrelevant
#     all_results.sort(key=lambda x: (-_r2(x["final_score"]), x["candidate"]["filename"]))

#     return all_results[:top_n]



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

FIXES (v2):
  - fuzzy_match now uses word-boundary matching instead of substring
  - LLM score is dominant when it signals irrelevance (< 0.25 → hard gate)
  - Blend formula is LLM-dominant for low-score candidates
  - Domain-aware relevance gate in rerank_and_explain
  - Semantic score is penalized for cross-domain mismatches via domain token check
  - Irrelevant candidates are filtered BEFORE returning results
"""

import hashlib
import numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ── Singleton embeddings ──────────────────────────────────────────────────────
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


# ── BM25 tokenizer ────────────────────────────────────────────────────────────
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
             "  \"domain_keywords\":  [\"keyword\", ...],\n"
             "  \"exp_years\":        <float>,\n"
             "  \"role_summary\":     \"<1-2 sentences>\"\n"
             "}}\n\n"
             "Rules:\n"
             "• required_skills  — must-have / required / essential\n"
             "• preferred_skills — nice-to-have / preferred / bonus\n"
             "• all_skills       — union of everything mentioned (hard + soft + domain)\n"
             "• domain_keywords  — 5-10 core domain words that DISTINGUISH this role from other domains.\n"
             "  For ML/AI roles: [\"machine learning\", \"model training\", \"neural network\", ...]\n"
             "  For sales roles: [\"quota\", \"pipeline\", \"b2b\", \"crm\", \"deal\", ...]\n"
             "  For HR roles:    [\"recruitment\", \"onboarding\", \"hris\", \"talent\", ...]\n"
             "  These are used to FILTER OUT candidates from completely unrelated domains.\n"
             "• Include domain skills too: 'financial modeling', 'stakeholder management',\n"
             "  'content strategy', 'client communication', 'agile delivery', etc.\n"
             "• skill_synonyms   — for each skill in all_skills, list common aliases,\n"
             "  abbreviations, and alternate forms.\n"
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
            "domain_keywords":  clean(result.get("domain_keywords",  [])),
            "exp_years":        float(result.get("exp_years", 0.0)),
            "role_summary":     result.get("role_summary", ""),
            "raw":              jd_text,
        }
    except Exception as exc:
        print(f"[retrieval_engine] JD parse failed: {exc} — semantic-only mode")
        return {
            "required_skills": [], "preferred_skills": [], "all_skills": [],
            "domain_keywords": [], "exp_years": 0.0, "role_summary": "", "raw": jd_text,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX — API CALL #2 happens here
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


# ── Dense retrieval ───────────────────────────────────────────────────────────
def semantic_top_k(index: ResumeIndex, jd_text: str, k: int) -> list:
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

SKILL_ALIASES = {
    # ── SQL / Databases ───────────────────────────────────────────────────────
    "sql":                    ["mysql", "postgresql", "postgres", "mssql", "sql server",
                               "oracle sql", "pl/sql", "sql queries", "sql plus", "t-sql",
                               "database queries", "rdbms", "relational database", "bigquery",
                               "redshift", "snowflake", "sqlite", "mariadb", "db2",
                               "sql scripting", "stored procedures", "sql optimization"],
    "nosql":                  ["mongodb", "cassandra", "dynamodb", "couchdb", "redis",
                               "elasticsearch", "firebase", "hbase", "neo4j", "couchbase"],
    "mysql":                  ["sql", "relational database", "rdbms", "mysql workbench"],
    "postgresql":             ["postgres", "sql", "rdbms", "pg", "pgadmin"],
    "mongodb":                ["nosql", "document database", "mongo"],
    "redis":                  ["in-memory database", "caching", "redis cache", "memcached"],
    "elasticsearch":          ["elastic search", "elk stack", "opensearch", "kibana"],

    # ── Programming Languages ─────────────────────────────────────────────────
    "python":                 ["python 3", "python3", "python 2", "py", "python scripting",
                               "python programming", "python development"],
    "java":                   ["core java", "java 8", "java 8+", "java 11", "java 17",
                               "java ee", "java se", "j2ee", "jvm", "java programming",
                               "java development", "object oriented java", "javascript"],
    "javascript":             ["js", "es6", "es6+", "es2015", "es2020", "vanilla js",
                               "node.js", "nodejs", "node js", "typescript", "ts", "jquery",
                               "javascript programming"],
    "typescript":             ["ts", "javascript", "js", "typed javascript"],
    "c#":                     ["csharp", "c sharp", ".net", "dotnet", "asp.net", "net core"],
    "c++":                    ["cpp", "c plus plus", "c/c++"],
    "r":                      ["r programming", "r language", "r studio", "rstudio"],
    "scala":                  ["apache scala", "scala programming"],
    "go":                     ["golang", "go programming", "go lang"],
    "ruby":                   ["ruby on rails", "rails", "ror"],
    "php":                    ["laravel", "symfony", "codeigniter", "php programming"],
    "swift":                  ["ios development", "xcode", "ios programming"],
    "kotlin":                 ["android development", "android programming"],
    "rust":                   ["rust programming", "rust lang"],
    "perl":                   ["perl scripting"],
    "shell":                  ["bash", "shell scripting", "bash scripting", "unix shell",
                               "linux scripting", "powershell"],

    # ── Backend Frameworks ────────────────────────────────────────────────────
    "spring boot":            ["spring", "spring mvc", "spring framework", "spring cloud",
                               "spring security", "spring data", "spring batch",
                               "spring boot framework", "spring rest"],
    "django":                 ["django rest framework", "drf", "django framework"],
    "fastapi":                ["fast api", "fastapi framework", "python fastapi"],
    "flask":                  ["flask framework", "flask api", "python flask"],
    "express":                ["express.js", "expressjs", "node express", "express framework"],
    "laravel":                ["php laravel", "laravel framework"],
    "rails":                  ["ruby on rails", "ror", "rails framework"],
    "asp.net":                ["asp net", "dotnet", ".net core", "asp.net core", "mvc .net"],
    "rest apis":              ["restful apis", "rest api", "restful services", "web services",
                               "rest", "api development", "rest endpoints", "http apis",
                               "restful web services", "api integration", "api design",
                               "rest architecture", "http rest", "json api", "xml api"],
    "graphql":                ["graph ql", "apollo graphql", "graphql api"],
    "microservices":          ["microservices architecture", "micro services",
                               "distributed systems", "service oriented architecture", "soa",
                               "event driven architecture", "service mesh", "api gateway"],
    "hibernate":              ["jpa", "hibernate jpa", "orm", "object relational mapping",
                               "hibernate/jpa", "hibernate/JPA", "spring data jpa",
                               "entity framework", "sequelize", "sqlalchemy"],
    "grpc":                   ["grpc framework", "protocol buffers", "protobuf"],
    "soap":                   ["soap web services", "wsdl", "xml web services"],

    # ── Frontend ──────────────────────────────────────────────────────────────
    "react":                  ["react.js", "reactjs", "react js", "react hooks",
                               "react native", "next.js", "nextjs", "react redux",
                               "react framework", "jsx", "react components"],
    "angular":                ["angularjs", "angular js", "angular 2+", "angular framework",
                               "angular cli", "ngrx"],
    "vue":                    ["vue.js", "vuejs", "nuxt.js", "vue framework", "vuex"],
    "html":                   ["html5", "html/css", "markup", "html templates", "html coding"],
    "css":                    ["css3", "bootstrap", "tailwind", "tailwind css",
                               "sass", "scss", "less", "styled components", "material ui",
                               "ant design", "chakra ui", "css frameworks", "responsive design",
                               "flexbox", "css grid"],
    "javascript":             ["js", "es6", "es6+", "typescript", "ts", "jquery",
                               "vanilla javascript", "frontend scripting", "java"],
    "redux":                  ["react redux", "state management", "flux", "mobx", "zustand"],
    "webpack":                ["bundler", "vite", "parcel", "rollup", "build tools frontend"],

    # ── Cloud Platforms ───────────────────────────────────────────────────────
    "aws":                    ["amazon web services", "ec2", "s3", "lambda", "rds",
                               "cloudwatch", "sagemaker", "eks", "ecs", "iam",
                               "route53", "cloudfront", "elastic beanstalk", "aws cloud",
                               "dynamodb", "sns", "sqs", "api gateway aws",
                               "aws bedrock", "step functions", "aws glue"],
    "azure":                  ["microsoft azure", "azure devops", "azure openai",
                               "azure functions", "azure blob", "azure aks",
                               "azure cosmos db", "azure service bus", "azure ad",
                               "azure storage", "azure pipelines"],
    "gcp":                    ["google cloud", "google cloud platform", "bigquery",
                               "google cloud storage", "gke", "cloud run",
                               "cloud functions", "vertex ai", "google cloud ai"],
    "cloud exposure":         ["aws", "azure", "gcp", "google cloud", "cloud platforms",
                               "cloud", "cloud computing", "cloud services",
                               "cloud deployment", "cloud infrastructure", "cloud native",
                               "multi cloud", "hybrid cloud"],

    # ── DevOps / CI-CD ────────────────────────────────────────────────────────
    "docker":                 ["containerization", "containers", "dockerized",
                               "docker compose", "dockerfile", "container management",
                               "docker swarm"],
    "kubernetes":             ["k8s", "container orchestration", "eks", "aks", "gke",
                               "helm", "kubectl", "openshift", "k8s cluster"],
    "jenkins":                ["ci/cd", "ci/cd pipelines", "continuous integration",
                               "continuous deployment", "pipeline", "jenkins pipeline"],
    "github actions":         ["ci/cd", "ci/cd pipelines", "github ci", "github workflows"],
    "gitlab ci":              ["gitlab pipelines", "ci/cd", "gitlab devops"],
    "ci/cd pipelines":        ["jenkins", "github actions", "azure devops", "gitlab ci",
                               "circleci", "travis ci", "continuous integration",
                               "continuous deployment", "devops pipeline", "build pipeline",
                               "deployment pipeline", "automated deployment"],
    "git":                    ["git/github", "github", "gitlab", "bitbucket",
                               "version control", "source control", "svn",
                               "git commands", "branching strategy", "git flow"],
    "terraform":              ["infrastructure as code", "iac", "pulumi", "cloudformation",
                               "infrastructure automation"],
    "ansible":                ["configuration management", "puppet", "chef",
                               "infrastructure automation"],
    "prometheus":             ["monitoring", "grafana", "alertmanager", "observability"],
    "grafana":                ["monitoring", "prometheus", "dashboards", "observability"],

    # ── Build Tools ───────────────────────────────────────────────────────────
    "maven":                  ["gradle", "build tools", "ant", "npm", "yarn", "pip",
                               "maven build", "pom.xml"],
    "gradle":                 ["maven", "build automation", "build tools"],
    "npm":                    ["node package manager", "yarn", "package manager", "pnpm"],

    # ── Agile / Project Management ────────────────────────────────────────────
    "agile":                  ["agile methodology", "agile/scrum", "scrum", "sprint",
                               "kanban", "agile development", "iterative development",
                               "agile practices", "agile ceremonies", "agile teams"],
    "scrum":                  ["agile/scrum", "agile", "sprint planning", "daily standup",
                               "sprint review", "retrospective", "scrum master",
                               "scrum ceremonies", "product backlog"],
    "jira":                   ["jira tool", "project management tool", "issue tracking",
                               "atlassian jira", "jira board", "jira tickets"],
    "confluence":             ["atlassian confluence", "documentation tool", "wiki"],

    # ── AI / ML ───────────────────────────────────────────────────────────────
    "machine learning":       ["ml", "scikit-learn", "sklearn", "supervised learning",
                               "unsupervised learning", "predictive modeling",
                               "classification", "regression", "ml models",
                               "model training", "feature engineering", "ml algorithms"],
    "deep learning":          ["neural networks", "cnn", "rnn", "lstm", "transformer",
                               "tensorflow", "pytorch", "keras", "dl",
                               "computer vision", "image classification"],
    "nlp":                    ["natural language processing", "text processing",
                               "text analytics", "sentiment analysis",
                               "named entity recognition", "text classification",
                               "language models", "text mining", "information extraction"],
    "large language models":  ["llm", "llms", "gpt", "openai", "claude", "gemini",
                               "llama", "mistral", "language models", "foundation models",
                               "gpt-4", "chatgpt", "generative ai", "llm development", "llm integration"],
    "rag pipelines":          ["retrieval augmented generation", "rag",
                               "retrieval-augmented generation", "semantic search",
                               "vector search", "knowledge retrieval", "document retrieval",
                               "rag system", "rag architecture"],
    "prompt engineering":     ["prompting", "prompt design", "prompt tuning",
                               "chain of thought", "few shot learning", "zero shot",
                               "prompt optimization", "llm prompting"],
    "embedding models":       ["embeddings", "text embeddings", "vector embeddings",
                               "sentence transformers", "word2vec", "openai embeddings",
                               "embedding generation", "dense embeddings"],
    "vector databases":       ["faiss", "pinecone", "chromadb", "weaviate", "qdrant",
                               "milvus", "vector store", "vector db", "vector index"],
    "agentic ai workflows":   ["ai agents", "autonomous agents", "agent workflows",
                               "tool use", "function calling", "tool calling",
                               "agent based systems", "llm agents"],
    "multi-agent systems":    ["multi agent", "crewai", "autogen", "langgraph",
                               "agent orchestration", "agent collaboration"],
    "mlops":                  ["ml operations", "model deployment", "model monitoring",
                               "mlflow", "kubeflow", "model serving", "model registry",
                               "feature store", "model pipeline"],
    "langchain":              ["lang chain", "langchain framework", "llm framework",
                               "langchain python"],
    "computer vision":        ["cv", "image processing", "opencv", "object detection",
                               "image recognition", "cnn", "yolo"],

    # ── Testing ───────────────────────────────────────────────────────────────
    "manual testing":         ["functional testing", "test execution", "black box testing",
                               "exploratory testing", "smoke testing", "sanity testing",
                               "regression testing", "uat", "user acceptance testing",
                               "system testing", "integration testing", "end to end testing"],
    "api testing":            ["rest api testing", "postman", "swagger", "swaggerui",
                               "api validation", "endpoint testing", "soap testing",
                               "api automation", "rest assured"],
    "automation testing":     ["selenium", "cypress", "playwright", "testng", "junit",
                               "pytest", "robot framework", "test automation",
                               "selenium webdriver", "automated testing", "ui automation"],
    "test case design":       ["test cases", "test case preparation", "test case development",
                               "test case writing", "test case execution", "test scenarios",
                               "test planning", "test strategy", "test design",
                               "test scenario creation", "test documentation"],
    "bug tracking":           ["defect tracking", "jira", "bugzilla", "hp alm",
                               "defect management", "defect lifecycle", "issue tracking",
                               "defect reporting", "bug reporting", "defect triage",
                               "bug management"],
    "database validation":    ["data validation", "backend data validation",
                               "sql validation", "data integrity", "db testing",
                               "data verification", "database testing"],
    "performance testing":    ["load testing", "stress testing", "jmeter",
                               "gatling", "locust", "k6", "performance benchmarking"],
    "security testing":       ["penetration testing", "pen testing", "vapt",
                               "vulnerability assessment", "owasp", "security scanning"],

    # ── Data Engineering ──────────────────────────────────────────────────────
    "airflow":                ["apache airflow", "workflow orchestration", "dag",
                               "data pipelines", "airflow dags"],
    "kafka":                  ["apache kafka", "event streaming", "message queue",
                               "rabbitmq", "activemq", "pubsub", "event driven",
                               "message broker", "streaming platform"],
    "spark":                  ["apache spark", "pyspark", "big data processing",
                               "spark streaming", "databricks"],
    "pandas":                 ["dataframes", "data manipulation", "numpy", "data analysis",
                               "data processing python"],
    "data pipelines":         ["etl", "elt", "data ingestion", "data processing",
                               "data integration", "data transformation", "data flow"],
    "tableau":                ["data visualization", "power bi", "looker", "metabase",
                               "business intelligence", "bi tools", "reporting tools"],
    "power bi":               ["powerbi", "microsoft power bi", "bi reporting",
                               "data visualization", "tableau"],

    # ── Communication / Soft Skills ───────────────────────────────────────────
    "communication skills":   ["collaboration", "cross-functional", "stakeholder management", "marketing communication"
                               "client communication", "team communication",
                               "interpersonal skills", "presentation skills",
                               "verbal communication", "written communication",
                               "cross functional teams", "team collaboration"],

    # ── Security ─────────────────────────────────────────────────────────────
    "jwt":                    ["json web token", "token based authentication", "oauth",
                               "oauth2", "authentication", "authorization",
                               "bearer token", "access token"],
    "spring security":        ["security", "rbac", "role based access control",
                               "authentication", "authorization", "spring auth"],
    "oauth":                  ["oauth2", "openid connect", "sso", "single sign on",
                               "identity provider", "keycloak"],

    # ── Networking / Infrastructure ───────────────────────────────────────────
    "linux":                  ["unix", "ubuntu", "centos", "rhel", "linux administration",
                               "linux commands", "bash", "shell"],
    "nginx":                  ["web server", "reverse proxy", "load balancer", "apache"],
    "rest":                   ["http", "https", "api", "web api", "rest architecture"],

    # ── Domain Specific ───────────────────────────────────────────────────────
    "banking domain":         ["fintech", "financial services", "banking", "finance domain",
                               "payment systems", "core banking", "investment banking",
                               "retail banking", "banking software", "financial technology",
                               "capital markets", "wealth management", "insurance",
                               "loan origination", "credit processing", "trade finance",
                               "risk management", "compliance", "regulatory reporting",
                               "swift", "sepa", "banking regulations", "basel"],
    "healthcare domain":      ["health tech", "medical software", "emr", "ehr",
                               "healthcare it", "hospital management", "clinical systems",
                               "patient management", "medical records", "hipaa",
                               "health informatics", "telemedicine", "medical devices",
                               "pharmacy systems", "laboratory systems", "radiology",
                               "healthcare analytics", "population health"],
    "ecommerce":              ["e-commerce", "online retail", "shopping platform",
                               "marketplace", "payment gateway", "order management",
                               "inventory management", "product catalog", "cart management",
                               "checkout", "payment processing", "shopify", "magento",
                               "woocommerce", "retail technology", "d2c", "b2c"],
    "erp":                    ["sap", "oracle erp", "enterprise resource planning",
                               "sap abap", "sap s4 hana", "sap hana", "oracle fusion",
                               "microsoft dynamics", "ms dynamics", "netsuite",
                               "odoo", "epicor", "infor", "sage erp"],
    "crm":                    ["salesforce", "hubspot", "customer relationship management",
                               "ms dynamics", "zoho crm", "pipedrive", "freshsales",
                               "salesforce crm", "crm software", "customer management",
                               "lead management", "sales automation", "crm platforms", "experience with crm platforms"],
    "crm proficiency":        ["hubspot", "hubspot crm", "salesforce", "zoho crm",
                               "crm tools", "crm software", "crm management",
                               "apollo.io", "pipedrive", "freshsales"],
    "education technology":   ["edtech", "lms", "learning management system", "e-learning",
                               "online learning", "student management", "school management",
                               "course management", "moodle", "canvas", "blackboard"],
    "telecom domain":         ["telecommunications", "telecom", "billing system",
                               "network management", "bss", "oss", "subscriber management",
                               "voice over ip", "voip", "5g", "network infrastructure",
                               "siebel crm", "amdocs", "telecom billing"],
    "logistics domain":       ["supply chain", "logistics", "warehouse management",
                               "fleet management", "transportation management",
                               "inventory tracking", "last mile delivery", "dispatch",
                               "freight management", "erp logistics"],
    "retail domain":          ["retail", "pos", "point of sale", "retail software",
                               "store management", "merchandising", "retail analytics",
                               "omnichannel", "inventory management"],
    "manufacturing domain":   ["manufacturing", "production planning", "quality control",
                               "shop floor", "scada", "plc", "iot manufacturing",
                               "lean manufacturing", "six sigma", "mes"],
    "legal domain":           ["legal tech", "legal software", "case management",
                               "contract management", "document management",
                               "compliance", "legal analytics", "law firm software"],
    "hr domain":              ["human resources", "hrms", "hris", "hr software",
                               "payroll", "recruitment", "talent management",
                               "performance management", "onboarding", "employee management",
                               "workday", "successfactors", "bamboohr"],
    "sales domain":           ["sales", "b2b sales", "b2c sales", "sales management",
                               "lead generation", "pipeline management", "quota",
                               "account management", "deal closing", "revenue"],
    "proposal preparation": [
    "proposal handling",
    "proposal creation",
    "proposal drafting",
    "business proposal",
    "proposal development",
    "proposal writing",
    "rfp response",
    "rfq response",
    "pitch deck creation",
    "proposal & pitch deck creation",
    "quotation preparation",
    "sales proposal"
],
    "marketing domain":       ["digital marketing", "seo", "sem", "content marketing",
                               "social media marketing", "email marketing",
                               "marketing automation", "google analytics", "hubspot",
                               "campaign management", "brand management"],
    "iot":                    ["internet of things", "iot devices", "embedded systems",
                               "sensor data", "mqtt", "edge computing", "iot platform",
                               "connected devices", "smart devices"],
    "cybersecurity":          ["information security", "network security", "ethical hacking",
                               "penetration testing", "vulnerability assessment",
                               "security operations", "soc", "siem", "firewall",
                               "intrusion detection", "owasp", "iso 27001"],
    "data science":           ["data analysis", "data analytics", "statistical analysis",
                               "data visualization", "business intelligence",
                               "predictive analytics", "statistical modeling",
                               "data exploration", "eda", "hypothesis testing"],
    "devops":                 ["development operations", "site reliability engineering",
                               "sre", "platform engineering", "infrastructure automation",
                               "cloud operations", "devsecops", "gitops"],
    "embedded systems":       ["firmware", "rtos", "real time systems", "microcontroller",
                               "arduino", "raspberry pi", "embedded c", "fpga"],
    "gaming":                 ["game development", "unity", "unreal engine", "game design",
                               "game programming", "3d development", "vr development",
                               "ar development"],
    "blockchain":             ["web3", "smart contracts", "solidity", "ethereum",
                               "distributed ledger", "defi", "nft", "cryptocurrency"],
    "negotiation":            ["deal negotiation","contract negotiation","commercial negotiation"],

    "closing":                    ["closure","deal closure","deal closing","sales closure"],
    "market research":            ["market trends","market analysis","industry analysis",
                               "competitive analysis","researching market"],
      
   
}

def normalize_skill(skill: str) -> str:
    """Return canonical form of a skill."""
    s = skill.strip().lower()
    for canonical, aliases in SKILL_ALIASES.items():
        if s in aliases:
            return canonical
    return s


# ── FIX #1: Stricter fuzzy_match — word-boundary only, no raw substring ───────
def fuzzy_match(jd_skill: str, cand_set: set, skill_synonyms: dict = None) -> bool:
    """
    Match a JD skill against a candidate's skill set.

    Rules (v2 — stricter):
    - Exact match always wins.
    - Single-word JD skill: must match as a whole word in candidate skill,
      not as a substring. "python" matches "python" and "python 3" but
      NOT "cpython" or "monopython".
    - Multi-word JD skill: all words must appear in candidate skill token set.
    - Synonyms are checked after the above.
    """
    if jd_skill in cand_set:
        return True

    jd_words = set(jd_skill.split())

    for cs in cand_set:
        cs_words = set(cs.split())

        if len(jd_words) == 1:
            # Word-boundary match: jd_skill must be a whole word token in cs
            # e.g. "python" matches "python 3" but not "cpython"
            if jd_skill in cs_words:
                return True
        else:
            # Multi-word: all JD skill words must appear in candidate skill
            if jd_words == cs_words:
                return True
            if jd_words.issubset(cs_words) and len(jd_words) >= 2:
                return True
            if cs_words.issubset(jd_words) and len(cs_words) >= 2:
                return True

    if skill_synonyms:
        for syn in skill_synonyms.get(jd_skill, []):
            syn_lower = syn.lower()
            syn_words = set(syn_lower.split())
            if syn_lower in cand_set:
                return True
            for cs in cand_set:
                cs_words = set(cs.split())
                if syn_words == cs_words:
                    return True
                if syn_words.issubset(cs_words) and len(syn_words) >= 2:
                    return True

    # Hyphen normalization: "scikit-learn" == "scikit learn"
    jd_nohyphen = jd_skill.replace("-", " ").replace("/", " ")
    for cs in cand_set:
        cs_nohyphen = cs.replace("-", " ").replace("/", " ")
        if jd_nohyphen == cs_nohyphen:
            return True
    # Version suffix normalization: "css" matches "css3", "html" matches "html5"
    for cs in cand_set:
        cs_base = cs.rstrip("0123456789")
        jd_base = jd_skill.rstrip("0123456789")
        if cs_base == jd_base:
            return True
    # Normalization match via SKILL_ALIASES
    jd_normalized = normalize_skill(jd_skill)
    for cs in cand_set:
        if normalize_skill(cs) == jd_normalized:
            return True
    # Check SKILL_ALIASES directly
    if jd_skill in SKILL_ALIASES:
        for alias in SKILL_ALIASES[jd_skill]:
            if alias in cand_set:
                return True
    # Check if candidate skill is alias of jd_skill
    for canonical, aliases in SKILL_ALIASES.items():
        if jd_skill in aliases or jd_skill == canonical:
            if canonical in cand_set or any(a in cand_set for a in aliases):
                return True

    return False


# ── FIX #2: Domain relevance pre-check (no API) ──────────────────────────────
def domain_relevance_score(candidate: dict, jd_features: dict) -> float:
    """
    Quick heuristic: how many domain_keywords appear in the candidate's raw text?
    Returns 0.0–1.0. Below 0.15 = likely wrong domain.
    This is a PRE-FILTER signal, not the main score.
    """
    domain_kws = jd_features.get("domain_keywords", [])
    if not domain_kws:
        return 1.0  # no domain info, don't filter

    raw = candidate.get("raw_text", "").lower()
    skills = set(candidate.get("skills", []))
    matches = 0
    for kw in domain_kws:
        kw_lower = kw.lower()
        kw_words = set(kw_lower.split())
        # Check raw text word-by-word
        raw_tokens = set(raw.split())
        if kw_words.issubset(raw_tokens):
            matches += 1
        elif any(kw_words.issubset(set(s.split())) for s in skills):
            matches += 1

    return matches / len(domain_kws)


def skill_overlap_score(candidate_skills: list, jd_features: dict) -> float:
    all_jd = set(jd_features.get("all_skills", []))
    if not all_jd:
        return 0.0
    cand_set   = set(s.lower() for s in candidate_skills)  # ← yeh badla
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
    cand_set = set(s.lower() for s in candidate.get("skills", []))
    jd_skills = set(jd_features.get("all_skills", []))
    req       = set(jd_features.get("required_skills", []))
    raw_tokens = set(tokenize(candidate.get("raw_text", "")))
    for jd_skill in jd_skills:
        if all(w in raw_tokens for w in jd_skill.split()):
            cand_set.add(jd_skill)

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
        "strengths":        [],
        "gaps":             [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API CALL #3 — Rerank + Explain all top-N candidates in ONE prompt
# ═══════════════════════════════════════════════════════════════════════════════

# FIX #3: Relevance gate threshold — LLM score below this = irrelevant candidate
LLM_RELEVANCE_GATE = 0.15

def rerank_and_explain(
    jd_text: str,
    jd_features: dict,
    candidates_pool: list,
    top_n: int,
    api_key: str,
) -> list:
    """
    Single GPT-4o-mini call — reranks + explains all candidates.

    FIX v2 changes:
    - LLM score < LLM_RELEVANCE_GATE triggers hard rejection (score → 0)
    - Blend is now LLM-dominant (0.75 llm / 0.25 orig) instead of 60/40
    - Irrelevant candidates are removed from results entirely
    - Domain relevance pre-filter applied before sending to LLM
    """
    if not candidates_pool or not api_key:
        return candidates_pool[:top_n]

    # Pre-filter: drop obvious domain mismatches before sending to LLM
    # This saves tokens and prevents LLM from being confused by irrelevant resumes
    domain_filtered = []
    for item in candidates_pool:
        dr = domain_relevance_score(item["candidate"], jd_features)
        item["domain_relevance"] = dr
        domain_filtered.append(item)

    # Sort so highest domain-relevance goes first (better LLM context)
    domain_filtered.sort(key=lambda x: (-x["domain_relevance"], -x["final_score"]))

    summaries = []
    for i, item in enumerate(domain_filtered):
        c       = item["candidate"]
        skills  = ", ".join(c.get("skills", [])[:20]) or "none listed"
        exp     = c.get("experience_years", 0)
        name    = c.get("name", f"Candidate {i}")
        ex      = item["explanation"]
        cand_skills_lower = set(s.lower() for s in c.get("skills", []))
        jd_all = set(jd_features.get("all_skills", []))
        syns = jd_features.get("skill_synonyms", {})
        raw_tokens = set(tokenize(c.get("raw_text", "")))
        for jd_skill in jd_all:
            if all(w in raw_tokens for w in jd_skill.split()):
                cand_skills_lower.add(jd_skill)
        matched_now = [s for s in jd_all if fuzzy_match(s, cand_skills_lower, syns)]
        matched = ", ".join(matched_now[:15]) or "none"
        chunks  = c.get("chunks", {})
        context = (
            chunks.get("skills", "") + " " +
            chunks.get("summary", "") + " " +
            chunks.get("experience", "")[:500]
        )[:900]
        dr_pct  = round(item.get("domain_relevance", 1.0) * 100)
        summaries.append(
            f"[{i}] {name} | {exp} yrs exp | domain_match={dr_pct}%\n"
            f"    ALREADY_MATCHED (NEVER list in gaps): {matched}\n"
            f"    context: {context}"
        )

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            max_tokens=min(8000, 1000 + 600 * len(domain_filtered)),
            # model_kwargs={"seed": 42},
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert recruiter. Evaluate candidates for the given role. "
             "Output ONLY valid JSON — no markdown, no extra text."),
            ("human",
             "Role: {role_summary}\n"
             "JD (first 800 chars): {jd_text}\n"
             "Required skills: {required_skills}\n"
             "Domain keywords (core to this role): {domain_keywords}\n\n"
             "Candidates:\n{summaries}\n\n"
             "SCORING RULES:\n"
             "1. Score 0.0–1.0 reflecting TRUE fit for this SPECIFIC role.\n"
             "2. A candidate from a completely different domain (e.g. QA engineer\n"
             "   for an ML role, cafe manager for a sales role) must score < 0.20\n"
             "   even if they share some general skills like 'python' or 'excel'.\n"
             "3. Generic skills (excel, python, communication) shared across domains\n"
             "   do NOT make someone relevant to this role. Core domain skills do.\n"
             "4. ALREADY_MATCHED skills are 100% confirmed present in resume — STRICTLY NEVER mention them in gaps or strengths contradictions. This is a HARD rule with no exceptions.\n"
             "5. Infer soft skills from experience context, don't hallucinate them.\n"
             "6. Candidate experience years are EXACT — never contradict them in strengths or gaps.\n"
             "6. Penalize keyword stuffing or inconsistent domain profiles.\n"
             "7. Only mention strengths explicitly supported by candidate context.\n"
             "8. Gaps ONLY from these JD required skills: {required_skills}. Never invent gaps not in JD.\n\n"
             "For EACH candidate return:\n"
             "  score     — float 0.0–1.0 (true fit for this specific role)\n"
             "  strengths — list of 2–3 strings, each under 12 words, specific\n"
             "  gaps      — list of 1–2 strings, each under 12 words, specific, ONLY from JD required skills that are genuinely missing\n"
             "  relevant  — true/false (is this candidate in the right domain?)\n\n"
             "Return ONLY this JSON:\n"
             "{{\"results\": [\n"
             "  {{\"score\": 0.92, \"strengths\": [\"...\"], \"gaps\": [\"...\"], \"relevant\": true}},\n"
             "  ...\n"
             "]}}"),
        ])
        chain  = prompt | llm | JsonOutputParser()

        # ── API CALL #3 ────────────────────────────────────────────────────────
        output = chain.invoke({
            "role_summary":     jd_features.get("role_summary", ""),
            "jd_text":          jd_text[:800],
            "required_skills":  ", ".join(jd_features.get("required_skills", [])[:12]),
            "domain_keywords":  ", ".join(jd_features.get("domain_keywords", [])[:10]),
            "summaries":        "\n".join(summaries),
        })
        # ──────────────────────────────────────────────────────────────────────

        llm_results = output.get("results", [])

        for i, item in enumerate(domain_filtered):
            if i >= len(llm_results):
                item["ce_score"] = 0.0
                item["is_relevant"] = True
                ex = item["explanation"]
                c  = item["candidate"]
                if not ex["strengths"] and ex["required_matched"]:
                    ex["strengths"].append(f"Meets: {', '.join(ex['required_matched'][:3])}")
                if not ex["gaps"] and ex["required_missing"]:
                    ex["gaps"].append(f"Missing: {', '.join(ex['required_missing'][:2])}")
                continue
            res   = llm_results[i]
            llm_s = max(0.0, min(1.0, float(res.get("score", 0.5))))
            is_relevant = res.get("relevant", True)
            orig  = item["final_score"]

            # ── FIX #3: Hard relevance gate ───────────────────────────────────
            # If LLM says score is very low OR explicitly marks as irrelevant,
            # zero out the score so it sinks to the bottom / gets filtered.
            if llm_s < LLM_RELEVANCE_GATE:
                blend = max(llm_s * 0.1, orig * 0.2)  # pre-score ka 20% bachao
            # ── FIX #4: LLM-dominant blend for low-mid scores ─────────────────
            elif llm_s < 0.5:
                # LLM is skeptical — balance with pre-score
                blend = 0.50 * llm_s + 0.50 * orig
            else:
                # LLM is confident — standard blend
                blend = 0.50 * llm_s + 0.50 * orig

            item["ce_score"]    = llm_s
            item["final_score"] = round(min(blend, 1.0), 4)
            item["explanation"]["score_pct"]  = round(min(blend, 1.0) * 100, 1)
            item["explanation"]["strengths"]  = res.get("strengths", [])[:4]
            item["explanation"]["gaps"] = res.get("gaps", [])[:3]
            item["is_relevant"] = is_relevant and llm_s >= LLM_RELEVANCE_GATE

            # ── Recompute matched/missing so they are consistent ──────────────
            c         = item["candidate"]
            cand_set  = set(s.lower() for s in c.get("skills", []))
            jd_skills = set(jd_features.get("all_skills", []))
            req       = set(jd_features.get("required_skills", []))
            syns      = jd_features.get("skill_synonyms", {})
            raw_tokens = set(tokenize(c.get("raw_text", "")))
            for jd_skill in jd_skills:
                if all(w in raw_tokens for w in jd_skill.split()):
                    cand_set.add(jd_skill)

            matched  = sorted(s for s in jd_skills if fuzzy_match(s, cand_set, syns))
            missing  = sorted(s for s in jd_skills if not fuzzy_match(s, cand_set, syns))

            # Extra safety: strip anything in matched from missing
            matched_set = set(matched)
            missing = [s for s in missing if s not in matched_set]

            item["explanation"]["matched_skills"]   = matched
            item["explanation"]["missing_skills"]   = missing[:8]
            item["explanation"]["required_matched"] = sorted(s for s in req if fuzzy_match(s, cand_set, syns))
            item["explanation"]["required_missing"] = sorted(s for s in req if not fuzzy_match(s, cand_set, syns))

            # Remove any matched skills from LLM gaps
            llm_gaps = item["explanation"]["gaps"]
            matched_words = set(" ".join(matched).lower().split())
            filtered_gaps = []
            for gap in llm_gaps:
                gap_words = set(gap.lower().split())
                if not gap_words.intersection(matched_words):
                    filtered_gaps.append(gap)
            item["explanation"]["gaps"] = filtered_gaps

    except Exception as exc:
        print(f"[retrieval_engine] Rerank+explain failed: {exc} — using pre-ranked scores")
        for item in domain_filtered:
            ex = item["explanation"]
            c  = item["candidate"]
            item["is_relevant"] = True  # don't filter on fallback
            if not ex["strengths"]:
                if ex["required_matched"]:
                    ex["strengths"].append(f"Meets required: {', '.join(ex['required_matched'][:3])}")
                if c.get("experience_years", 0) > 0:
                    ex["strengths"].append(f"{c['experience_years']} years of experience")
                if ex["matched_skills"]:
                    ex["strengths"].append(f"{len(ex['matched_skills'])} JD skills matched")
            if not ex["gaps"] and ex["required_missing"]:
                ex["gaps"].append(f"Missing required: {', '.join(ex['required_missing'][:3])}")

    # Sort and filter
    domain_filtered.sort(key=lambda x: (-x["final_score"], x["candidate"]["filename"]))

    # Remove clearly irrelevant candidates from results
    filtered = [r for r in domain_filtered if r.get("is_relevant", True) or r["final_score"] > 0.15]

    return domain_filtered[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def retrieve_top_n(
    index:    ResumeIndex,
    jd_text:  str,
    top_n:    int,
    api_key:  str,
) -> list:
    if not index.is_ready():
        return []


    # ── Call #1 ───────────────────────────────────────────────────────────────
    jd_hash = hashlib.md5(jd_text.encode()).hexdigest()
    if not hasattr(retrieve_top_n, "_jd_cache"):
        retrieve_top_n._jd_cache = {}
    if jd_hash in retrieve_top_n._jd_cache:
        jd_features = retrieve_top_n._jd_cache[jd_hash]
    else:
        jd_features = extract_jd_features(jd_text, api_key)
        retrieve_top_n._jd_cache[jd_hash] = jd_features
    print("ALL SKILLS:", jd_features["all_skills"])
    for c in index.candidates:
        print("NAME:", c.get("name"), "| exp:", c.get("experience_years"))
    jd_tokens = tokenize(jd_text)

    # ── Retrieval (no new LLM call) ───────────────────────────────────────────
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

        # FIX #5: Penalize pre-score using domain relevance signal
        domain_rel = domain_relevance_score(candidate, jd_features)
        domain_penalty = 1.0 if domain_rel >= 0.1 else (0.7 + domain_rel * 3.0)

        # final_score = min(
        #     (0.45 * skill_sc + 0.25 * sem_adj + 0.25 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0))
        #     * domain_penalty,
        #     1.0,
        # )

        final_score = min(
            (0.40 * skill_sc + 0.20 * sem_adj + 0.35 * exp_sc + 0.05 * min(skill_sc * 1.1, 1.0))
            * domain_penalty,
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
    rerank_pool = results[: min(top_n * 2, 15)]

    # ── Call #3 ───────────────────────────────────────────────────────────────
    return rerank_and_explain(jd_text, jd_features, rerank_pool, top_n, api_key)