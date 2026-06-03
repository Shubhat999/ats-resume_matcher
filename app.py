# This is final 
# """
# AI Resume Screener — Streamlit App (LangChain · 3 API Calls)
# ══════════════════════════════════════════════════════════════════════════════
# API call budget per analysis run:
#   #1  extract_jd_features   — gpt-4o-mini      — JD → skills/exp/summary
#   #2  ResumeIndex.build()   — text-embedding-3-large — embed all resume chunks
#   #3  rerank_and_explain()  — gpt-4o-mini      — score + explain all top-N

# Upload time: 0 API calls (pure text extraction + heuristics).
# API key: loaded ONLY from .env / environment — never entered in UI.
# """

# import os
# import time
# import json
# import hashlib
# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# from dotenv import load_dotenv

# from resume_parser import parse_resume
# from retrieval_engine import ResumeIndex, retrieve_top_n, validate_jd

# load_dotenv()

# # ── Page config ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Resume Screener AI",
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Force sidebar always visible
# st.markdown("""
# <style>
# [data-testid="collapsedControl"] {
#     display: flex !important;
#     visibility: visible !important;
#     opacity: 1 !important;
# }
# section[data-testid="stSidebar"] {
#     display: flex !important;
#     visibility: visible !important;
#     transform: none !important;
#     min-width: 280px !important;
# }
# section[data-testid="stSidebar"][aria-expanded="false"] {
#     margin-left: 0 !important;
#     transform: none !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── CSS ────────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
# html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
# .stApp { background-color: #0d1117; color: #e6edf3; }
# section[data-testid="stSidebar"] {
#     background-color: #161b22 !important;
#     border-right: 1px solid #21262d;
#     display: flex !important;
#     visibility: visible !important;
#     transform: translateX(0) !important;
#     min-width: 244px !important;
#     max-width: 320px !important;
# }
# section[data-testid="stSidebar"][aria-expanded="false"] {
#     transform: translateX(0) !important;
#     min-width: 244px !important;
# }
# [data-testid="collapsedControl"] {
#     display: flex !important;
#     visibility: visible !important;
# }
# section[data-testid="stSidebar"] .stMarkdown p,
# section[data-testid="stSidebar"] label { color: #8b949e !important; }
# .app-header {
#     background: linear-gradient(135deg,#1a2332 0%,#0d1117 100%);
#     border:1px solid #21262d; border-radius:12px;
#     padding:24px 32px; margin-bottom:24px;
#     display:flex; align-items:center; gap:16px;
# }
# .app-header h1 { font-size:26px; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.5px; }
# .app-header p  { font-size:13px; color:#8b949e; margin:4px 0 0 0; }
# .badge {
#     background:#1f6feb22; border:1px solid #1f6feb55; color:#58a6ff;
#     font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
#     font-family:'DM Mono',monospace; letter-spacing:0.5px;
# }
# .stButton > button {
#     background:#238636 !important; color:white !important; border:none !important;
#     border-radius:8px !important; font-weight:600 !important; font-size:14px !important;
#     padding:10px 24px !important; width:100%; transition:background 0.2s !important;
# }
# .stButton > button:hover    { background:#2ea043 !important; }
# .stButton > button:disabled { background:#21262d !important; color:#484f58 !important; }
# .stMetric { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }
# .stTextArea textarea {
#     background-color:#161b22 !important; border:1px solid #30363d !important;
#     border-radius:8px !important; color:#e6edf3 !important; font-size:14px !important;
# }
# .stTextArea textarea:focus { border-color:#1f6feb !important; box-shadow:0 0 0 3px #1f6feb22 !important; }
# .info-box    { background:#1a2740; border:1px solid #1f4080; border-radius:8px; padding:12px 16px; font-size:13px; color:#79c0ff; margin:8px 0; }
# .success-box { background:#1a2e1a; border:1px solid #2d5a2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#56d364; margin:8px 0; }
# .warning-box { background:#2e2218; border:1px solid #6e4c1a; border-radius:8px; padding:12px 16px; font-size:13px; color:#e3b341; margin:8px 0; }
# .error-box   { background:#2e1a1a; border:1px solid #5a2d2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#f85149; margin:8px 0; }
# .resume-item {
#     background:#0d1117; border:1px solid #21262d; border-radius:6px;
#     padding:8px 12px; margin:4px 0; font-size:12px; color:#8b949e;
#     display:flex; align-items:center; gap:8px;
# }
# .resume-count-badge {
#     background:#1f6feb33; color:#58a6ff; border-radius:12px;
#     padding:2px 8px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace;
# }
# /* Always show sidebar collapse/expand arrow */
# button[kind="header"] { display: block !important; }
# [data-testid="collapsedControl"] { display: flex !important; }
                       
# #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
# ::-webkit-scrollbar { width:6px; }
# ::-webkit-scrollbar-track { background:#0d1117; }
# ::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
# hr { border-color:#21262d !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── Session state ──────────────────────────────────────────────────────────────
# for k, v in {
#     "resume_index":   ResumeIndex(),
#     "parsed_resumes": {},
#     "results":        [],
#     "last_jd_hash":   "",
#     "index_built":    False,
#     "last_elapsed":   0.0,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ── API key (env only) ────────────────────────────────────────────────────────
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def make_jd_hash(jd_text: str, resume_hashes: list, top_n: int) -> str:
#     combined = jd_text.strip() + "||" + ",".join(sorted(resume_hashes)) + f"||{top_n}"
#     return hashlib.sha256(combined.encode()).hexdigest()[:12]

# def score_color(s: float) -> str:
#     return "score-high" if s >= 0.70 else ("score-mid" if s >= 0.45 else "score-low")

# def bar_color(s: float) -> str:
#     return "#3fb950" if s >= 0.70 else ("#f0883e" if s >= 0.45 else "#f85149")

# def card_class(rank: int) -> str:
#     return {1: "top1", 2: "top2", 3: "top3"}.get(rank, "rest")

# def rebuild_index():
#     resumes = list(st.session_state.parsed_resumes.values())
#     if not resumes or not OPENAI_API_KEY:
#         return
#     idx = ResumeIndex()
#     idx.build(resumes, OPENAI_API_KEY)   # ← API Call #2
#     st.session_state.resume_index = idx
#     st.session_state.index_built  = True


# # ── Sidebar ────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style="padding:16px 0 8px 0;">
#         <div style="font-size:18px;font-weight:700;color:#e6edf3;">🎯 Resume Screener</div>
#         <div style="font-size:11px;color:#8b949e;margin-top:4px;">3 API Calls · Any Domain · LangChain</div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.divider()

#     # API key status
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
#     if OPENAI_API_KEY:
#         masked = OPENAI_API_KEY[:7] + "••••••••" + OPENAI_API_KEY[-4:]
#         st.markdown(
#             f'<div class="success-box">✓ Loaded from environment<br>'
#             f'<span style="font-family:DM Mono,monospace;font-size:11px;opacity:0.6;">{masked}</span></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="error-box">✗ OPENAI_API_KEY not found<br>'
#             '<span style="font-size:11px;">Add to .env or set as env variable</span></div>',
#             unsafe_allow_html=True,
#         )
#     st.divider()

#     # Top-N slider
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔢 Results to Show</div>', unsafe_allow_html=True)
#     top_n = st.slider("Top N", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")
#     st.markdown(f'<div style="font-size:11px;color:#8b949e;margin-bottom:4px;">Top <b style="color:#e6edf3">{top_n}</b> candidates</div>', unsafe_allow_html=True)
#     st.divider()

#     # Upload
#     api_ok = bool(OPENAI_API_KEY)
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">📤 Upload Resumes</div>', unsafe_allow_html=True)
#     if not api_ok:
#         st.markdown('<div class="warning-box">⚠ Set OPENAI_API_KEY first</div>', unsafe_allow_html=True)

#     uploaded_files = st.file_uploader(
#         "PDF / DOCX / TXT",
#         type=["pdf", "docx", "doc", "txt"],
#         accept_multiple_files=True,
#         label_visibility="collapsed",
#         disabled=not api_ok,
#     )

#     if uploaded_files and api_ok:
#         new_files = parse_errors = 0
#         progress  = st.progress(0)
#         for i, f in enumerate(uploaded_files):
#             fb = f.read()
#             h  = hashlib.sha256(fb).hexdigest()[:16]
#             if h not in st.session_state.parsed_resumes:
#                 parsed = parse_resume(fb, f.name)   # 0 API calls
#                 if parsed:
#                     st.session_state.parsed_resumes[h] = parsed
#                     new_files += 1
#                 else:
#                     parse_errors += 1
#             progress.progress((i + 1) / len(uploaded_files))
#         progress.empty()

#         if new_files > 0:
#             with st.spinner(f"Embedding {new_files} resume(s)…  [API call #2]"):
#                 rebuild_index()
#             st.session_state.last_jd_hash = ""
#             st.session_state.results = []
#             st.markdown(f'<div class="success-box">✓ {new_files} resume(s) indexed</div>', unsafe_allow_html=True)
#         if parse_errors:
#             st.markdown(f'<div class="warning-box">⚠ {parse_errors} file(s) failed to parse</div>', unsafe_allow_html=True)

#     st.divider()

#     # Resume list
#     total = len(st.session_state.parsed_resumes)
#     st.markdown(
#         f'<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
#         f'📁 Indexed <span class="resume-count-badge">{total}</span></div>',
#         unsafe_allow_html=True,
#     )
#     if total == 0:
#         st.markdown('<div style="font-size:12px;color:#484f58;padding:8px 0;">No resumes yet.</div>', unsafe_allow_html=True)
#     else:
#         for r in list(st.session_state.parsed_resumes.values())[:40]:
#             name = r.get("name", r["filename"])
#             exp  = r.get("experience_years", 0)
#             nsk  = len(r.get("skills", []))
#             st.markdown(
#                 f'<div class="resume-item">📄 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{r["filename"]}">'
#                 f'{name}</span><span style="color:#484f58;font-size:10px;">{exp}y·{nsk}sk</span></div>',
#                 unsafe_allow_html=True,
#             )
#         if total > 40:
#             st.markdown(f'<div style="font-size:11px;color:#484f58;">…and {total-40} more</div>', unsafe_allow_html=True)

#     st.divider()
#     if total > 0:
#         if st.button("🗑  Clear All", use_container_width=True):
#             st.session_state.parsed_resumes = {}
#             st.session_state.resume_index   = ResumeIndex()
#             st.session_state.results        = []
#             st.session_state.index_built    = False
#             st.rerun()

#     st.markdown("""
#     <div style="margin-top:auto;padding-top:20px;font-size:11px;color:#484f58;line-height:1.9;">
#     <b style="color:#30363d;">API calls/run:</b> exactly 3<br>
#     <b style="color:#30363d;">Upload calls:</b> 0 (free)<br>
#     <b style="color:#30363d;">Embeddings:</b> text-embedding-3-large<br>
#     <b style="color:#30363d;">LLM:</b> gpt-4o-mini<br>
#     <b style="color:#30363d;">Key:</b> env-only, never in UI
#     </div>
#     """, unsafe_allow_html=True)


# # ── Main area ──────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="app-header">
#     <div>
#         <h1>🎯 AI Resume Screener</h1>
#         <p>3 API calls per run · text-embedding-3-large · BM25 · RRF · LLM Rerank+Explain · Any domain</p>
#     </div>
#     <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
#         <span class="badge">3 API CALLS</span>
#         <span class="badge">text-embedding-3-large</span>
#         <span class="badge">BM25 + RRF</span>
#         <span class="badge">ANY DOMAIN</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# total_resumes = len(st.session_state.parsed_resumes)
# c1, c2, c3, c4 = st.columns(4)
# with c1: st.metric("Resumes Indexed", total_resumes)
# with c2:
#     avg = round(sum(len(r.get("skills",[])) for r in st.session_state.parsed_resumes.values()) / max(total_resumes,1), 1)
#     st.metric("Avg Skill Tokens", avg)
# with c3: st.metric("Top Results", top_n)
# with c4: st.metric("API Key", "✓ Set" if OPENAI_API_KEY else "✗ Missing")

# st.markdown("---")

# st.markdown('<div style="font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:8px;">📋 Job Description</div>', unsafe_allow_html=True)
# st.markdown('<div style="font-size:12px;color:#8b949e;margin-bottom:12px;">Paste any JD — tech, sales, HR, marketing, ops, finance. LLM understands it all.</div>', unsafe_allow_html=True)

# jd_text = st.text_area(
#     "JD",
#     height=220,
#     placeholder=(
#         "Tech:     Senior Python Engineer, 4+ yrs, FastAPI, PostgreSQL, Docker, RAG/LLM experience\n\n"
#         "Non-tech: Marketing Manager, 5+ yrs, brand strategy, SEO, Google Analytics, content creation\n\n"
#         "Sales:    Account Executive, SaaS B2B, Salesforce, quota achievement, enterprise deals"
#     ),
#     label_visibility="collapsed",
# )

# can_run = total_resumes > 0 and bool(jd_text.strip()) and api_ok
# col_btn, col_info = st.columns([1, 3])

# with col_btn:
#     analyze_btn = st.button("🔍  Analyze Candidates", use_container_width=True, disabled=not can_run)

# with col_info:
#     if not api_ok:
#         st.markdown('<div class="error-box">✗ Set OPENAI_API_KEY in .env to use this tool.</div>', unsafe_allow_html=True)
#     elif total_resumes == 0:
#         st.markdown('<div class="warning-box">⚠ Upload resumes first.</div>', unsafe_allow_html=True)
#     elif not jd_text.strip():
#         st.markdown('<div class="info-box">ℹ Paste a job description then click Analyze.</div>', unsafe_allow_html=True)
#     else:
#         jd_error = validate_jd(jd_text)
#         if jd_error:
#             st.markdown(f'<div class="warning-box">{jd_error}</div>', unsafe_allow_html=True)
#         else:
#             cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)
#             if cur_hash == st.session_state.last_jd_hash and st.session_state.results:
#                 st.markdown('<div class="success-box">✓ Cached — same JD + resumes + top-N → same results.</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="info-box">ℹ Ready · {total_resumes} resumes · top {top_n} · 3 API calls</div>', unsafe_allow_html=True)

# # ── Run ────────────────────────────────────────────────────────────────────────
# if analyze_btn and can_run:
#     jd_error = validate_jd(jd_text)
#     if jd_error:
#         st.markdown(f'<div class="error-box">{jd_error}</div>', unsafe_allow_html=True)
#         st.stop()
#     cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)

#     if cur_hash != st.session_state.last_jd_hash or not st.session_state.results:
#         if not st.session_state.index_built:
#             with st.spinner("Building embedding index… [API call #2]"):
#                 rebuild_index()

#         with st.spinner("Call #1: parsing JD  →  Call #3: reranking & generating explanations…"):
#             t0      = time.time()
#             results = retrieve_top_n(
#                 st.session_state.resume_index,
#                 jd_text,
#                 top_n=top_n,
#                 api_key=OPENAI_API_KEY,
#             )
#             elapsed = time.time() - t0

#         st.session_state.results      = results
#         st.session_state.last_jd_hash = cur_hash
#         st.session_state.last_elapsed = elapsed
#     else:
#         results = st.session_state.results


# # ── Results ────────────────────────────────────────────────────────────────────
# results = st.session_state.results

# if results:
#     elapsed = st.session_state.get("last_elapsed", 0)
#     st.markdown("---")
#     st.markdown(
#         f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
#         f'<div style="font-size:18px;font-weight:700;color:#e6edf3;">Top {len(results)} Candidates</div>'
#         f'<div style="font-size:12px;color:#8b949e;font-family:DM Mono,monospace;">'
#         f'{elapsed:.2f}s · 3 API calls · text-embedding-3-large + BM25 + RRF + LLM reranker</div>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 Summary Table", expanded=False):
#         rows = []
#         for rank, r in enumerate(results, 1):
#             c, ex = r["candidate"], r["explanation"]
#             rows.append({
#                 "Rank":           rank,
#                 "Name":           c.get("name", c["filename"]),
#                 "Match %":        f"{ex['score_pct']}%",
#                 "Exp (yrs)":      c.get("experience_years", "—"),
#                 "Skills Matched": len(ex["matched_skills"]),
#                 "Required Met":   len(ex["required_matched"]),
#                 "File":           c["filename"],
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

#     # Cards
#     CARD_CSS = """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
#     *{box-sizing:border-box;margin:0;padding:0}
#     body{background:transparent;font-family:'DM Sans',sans-serif}
#     .rank-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px 24px;margin-bottom:4px;position:relative}
#     .rank-card.top1{border-left:3px solid #f0883e}
#     .rank-card.top2{border-left:3px solid #58a6ff}
#     .rank-card.top3{border-left:3px solid #3fb950}
#     .rank-card.rest{border-left:3px solid #30363d}
#     .rank-number{position:absolute;top:16px;right:20px;font-size:28px;font-weight:700;color:#21262d;font-family:'DM Mono',monospace}
#     .candidate-name{font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:4px}
#     .candidate-meta{font-size:12px;color:#8b949e;margin-bottom:12px;font-family:'DM Mono',monospace}
#     .score-bar-wrap{background:#21262d;border-radius:4px;height:6px;margin:8px 0 4px;overflow:hidden}
#     .score-bar{height:100%;border-radius:4px}
#     .score-label{font-size:24px;font-weight:700;font-family:'DM Mono',monospace}
#     .score-high{color:#3fb950}.score-mid{color:#f0883e}.score-low{color:#f85149}
#     .tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:500;margin:2px 3px 2px 0;font-family:'DM Mono',monospace}
#     .tag-green{background:#1a2e1a;color:#3fb950;border:1px solid #2d5a2d}
#     .tag-red  {background:#2e1a1a;color:#f85149;border:1px solid #5a2d2d}
#     .tag-gray {background:#21262d;color:#8b949e;border:1px solid #30363d}
#     .sec-label{font-size:11px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px}
#     .str-item{font-size:13px;color:#3fb950;padding:2px 0}.str-item::before{content:"✓  "}
#     .gap-item{font-size:13px;color:#f85149;padding:2px 0}.gap-item::before{content:"✗  "}
#     .score-meta{font-size:10px;color:#484f58;font-family:'DM Mono',monospace;margin-top:2px}
#     </style>
#     """

#     html = CARD_CSS
#     for rank, r in enumerate(results, 1):
#         c, ex     = r["candidate"], r["explanation"]
#         score     = r["final_score"]
#         pct       = ex["score_pct"]
#         cc        = score_color(score)
#         bc        = bar_color(score)
#         cl        = card_class(rank)
#         medal     = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
#         exp_disp  = f"{c.get('experience_years',0)} yrs" if c.get("experience_years") else "—"
#         email     = c.get("email","")
#         sk_prev   = c.get("skills",[])[:8]
#         sk_pct    = round(r["skill_score"]*100)
#         sem_pct   = round(r["semantic_score"]*100)
#         ep        = round(r["experience_score"]*100)
#         llm_pct   = round(r.get("ce_score",0)*100)

#         matched_html = "".join(f'<span class="tag tag-green">{s}</span>' for s in ex["matched_skills"][:10]) \
#                     or '<span style="color:#484f58;font-size:12px;">None detected</span>'
#         miss_html    = (
#             '<div class="sec-label" style="margin-top:10px;">Missing Skills</div>'
#             + "".join(f'<span class="tag tag-red">{s}</span>' for s in ex["missing_skills"][:6])
#         ) if ex["missing_skills"] else ""
#         str_html     = "".join(f'<div class="str-item">{s}</div>' for s in ex["strengths"][:4]) \
#                     or '<div style="color:#484f58;font-size:12px;">—</div>'
#         gap_html     = '<div style="margin-top:6px;">' + "".join(f'<div class="gap-item">{g}</div>' for g in ex["gaps"][:3]) + "</div>" \
#                     if ex["gaps"] else ""
#         all_sk_html  = "".join(f'<span class="tag tag-gray">{s}</span>' for s in sk_prev)
#         extra        = len(c.get("skills",[])) - 8
#         if extra > 0:
#             all_sk_html += f'<span class="tag tag-gray">+{extra} more</span>'

#         html += f"""
#         <div class="rank-card {cl}">
#           <div class="rank-number">{medal}</div>
#           <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
#             <div style="flex:1;min-width:200px;">
#               <div class="candidate-name">{c.get("name", c["filename"])}</div>
#               <div class="candidate-meta">{c["filename"]} &middot; {exp_disp} &middot; {email}</div>
#               <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
#                 <div class="score-label {cc}">{pct}%</div>
#                 <div style="flex:1;">
#                   <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{bc};"></div></div>
#                   <div class="score-meta">Skills {sk_pct}% &middot; Semantic {sem_pct}% &middot; Exp {ep}% &middot; LLM {llm_pct}%</div>
#                 </div>
#               </div>
#             </div>
#           </div>
#           <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px;">
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">Matched Skills</div><div>{matched_html}</div>{miss_html}
#             </div>
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">Why Selected</div>{str_html}{gap_html}
#             </div>
#           </div>
#           <div style="margin-top:12px;">
#             <div class="sec-label">All Skill Tokens</div><div>{all_sk_html}</div>
#           </div>
#         </div>
#         """

#     components.html(html, height=340 * len(results) + 60, scrolling=False)

#     # Export
#     st.markdown("---")
#     export = []
#     for rank, r in enumerate(results, 1):
#         c, ex = r["candidate"], r["explanation"]
#         export.append({
#             "Rank":               rank,
#             "Name":               c.get("name",""),
#             "File":               c["filename"],
#             "Match_%":            ex["score_pct"],
#             "Experience_Years":   c.get("experience_years",0),
#             "Email":              c.get("email",""),
#             "Matched_Skills":     ", ".join(ex["matched_skills"]),
#             "Missing_Skills":     ", ".join(ex["missing_skills"]),
#             "Strengths":          " | ".join(ex["strengths"]),
#             "Gaps":               " | ".join(ex["gaps"]),
#             "Skill_Score_%":      round(r["skill_score"]*100,1),
#             "Semantic_Score_%":   round(r["semantic_score"]*100,1),
#             "Experience_Score_%": round(r["experience_score"]*100,1),
#             "LLM_Rerank_%":       round(r.get("ce_score",0)*100,1),
#         })

#     col1, col2 = st.columns(2)
#     with col1:
#         st.download_button("📥 Download CSV",  data=pd.DataFrame(export).to_csv(index=False), file_name="top_candidates.csv",  mime="text/csv",        use_container_width=True)
#     with col2:
#         st.download_button("📥 Download JSON", data=json.dumps(export, indent=2),             file_name="top_candidates.json", mime="application/json", use_container_width=True)

# elif total_resumes == 0:
#     st.markdown("""
#     <div style="text-align:center;padding:60px 24px;color:#484f58;">
#         <div style="font-size:48px;margin-bottom:16px;">📂</div>
#         <div style="font-size:18px;font-weight:600;color:#30363d;margin-bottom:8px;">No resumes uploaded</div>
#         <div style="font-size:13px;">Upload PDF, DOCX, or TXT files from the sidebar.</div>
#     </div>
#     """, unsafe_allow_html=True)



# """
# AI Resume Screener — Streamlit App (LangChain · 3 API Calls)
# ══════════════════════════════════════════════════════════════════════════════
# API call budget per analysis run:
#   #1  extract_jd_features   — gpt-4o-mini      — JD → skills/exp/summary
#   #2  ResumeIndex.build()   — text-embedding-3-large — embed all resume chunks
#   #3  rerank_and_explain()  — gpt-4o-mini      — LLM resolves skill matches,
#                                                   scores, and explains all top-N

# Upload time: 0 API calls (pure text extraction + heuristics).
# API key: loaded ONLY from .env / environment — never entered in UI.

# FIXES:
#   - Resume Tokens: tags had no spacing → "prasadprasad gaikwad" display bug
#     Fixed: added gap:6px + flex-wrap to token container, margin on each tag
#   - LLM Match %: was showing raw ce_score*100, now shows correctly with % label
#   - Score breakdown line: cleaner label "LLM {x}% · Embed {y}% · Exp {z}%"
#   - Fallback candidates (LLM Match 0%) now clearly marked as "Heuristic only"
#     in the score breakdown so user knows LLM didn't process them
# """

# import os
# import time
# import json
# import hashlib
# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# from dotenv import load_dotenv

# from resume_parser import parse_resume
# from retrieval_engine import ResumeIndex, retrieve_top_n

# load_dotenv()

# # ── Page config ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Resume Screener AI",
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── CSS ────────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
# html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
# .stApp { background-color: #0d1117; color: #e6edf3; }
# section[data-testid="stSidebar"] {
#     background-color: #161b22 !important; border-right: 1px solid #21262d;
# }
# section[data-testid="stSidebar"] .stMarkdown p,
# section[data-testid="stSidebar"] label { color: #8b949e !important; }
# .app-header {
#     background: linear-gradient(135deg,#1a2332 0%,#0d1117 100%);
#     border:1px solid #21262d; border-radius:12px;
#     padding:24px 32px; margin-bottom:24px;
#     display:flex; align-items:center; gap:16px;
# }
# .app-header h1 { font-size:26px; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.5px; }
# .app-header p  { font-size:13px; color:#8b949e; margin:4px 0 0 0; }
# .badge {
#     background:#1f6feb22; border:1px solid #1f6feb55; color:#58a6ff;
#     font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
#     font-family:'DM Mono',monospace; letter-spacing:0.5px;
# }
# .stButton > button {
#     background:#238636 !important; color:white !important; border:none !important;
#     border-radius:8px !important; font-weight:600 !important; font-size:14px !important;
#     padding:10px 24px !important; width:100%; transition:background 0.2s !important;
# }
# .stButton > button:hover    { background:#2ea043 !important; }
# .stButton > button:disabled { background:#21262d !important; color:#484f58 !important; }
# .stMetric { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }
# .stTextArea textarea {
#     background-color:#161b22 !important; border:1px solid #30363d !important;
#     border-radius:8px !important; color:#e6edf3 !important; font-size:14px !important;
# }
# .stTextArea textarea:focus { border-color:#1f6feb !important; box-shadow:0 0 0 3px #1f6feb22 !important; }
# .info-box    { background:#1a2740; border:1px solid #1f4080; border-radius:8px; padding:12px 16px; font-size:13px; color:#79c0ff; margin:8px 0; }
# .success-box { background:#1a2e1a; border:1px solid #2d5a2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#56d364; margin:8px 0; }
# .warning-box { background:#2e2218; border:1px solid #6e4c1a; border-radius:8px; padding:12px 16px; font-size:13px; color:#e3b341; margin:8px 0; }
# .error-box   { background:#2e1a1a; border:1px solid #5a2d2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#f85149; margin:8px 0; }
# .resume-item {
#     background:#0d1117; border:1px solid #21262d; border-radius:6px;
#     padding:8px 12px; margin:4px 0; font-size:12px; color:#8b949e;
#     display:flex; align-items:center; gap:8px;
# }
# .resume-count-badge {
#     background:#1f6feb33; color:#58a6ff; border-radius:12px;
#     padding:2px 8px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace;
# }
# #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
# ::-webkit-scrollbar { width:6px; }
# ::-webkit-scrollbar-track { background:#0d1117; }
# ::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
# hr { border-color:#21262d !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── Session state ──────────────────────────────────────────────────────────────
# for k, v in {
#     "resume_index":   ResumeIndex(),
#     "parsed_resumes": {},
#     "results":        [],
#     "last_jd_hash":   "",
#     "index_built":    False,
#     "last_elapsed":   0.0,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ── API key (env only) ────────────────────────────────────────────────────────
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def make_jd_hash(jd_text: str, resume_hashes: list, top_n: int) -> str:
#     combined = jd_text.strip() + "||" + ",".join(sorted(resume_hashes)) + f"||{top_n}"
#     return hashlib.sha256(combined.encode()).hexdigest()[:12]

# def score_color(s: float) -> str:
#     return "score-high" if s >= 0.70 else ("score-mid" if s >= 0.45 else "score-low")

# def bar_color(s: float) -> str:
#     return "#3fb950" if s >= 0.70 else ("#f0883e" if s >= 0.45 else "#f85149")

# def card_class(rank: int) -> str:
#     return {1: "top1", 2: "top2", 3: "top3"}.get(rank, "rest")

# def rebuild_index():
#     resumes = list(st.session_state.parsed_resumes.values())
#     if not resumes or not OPENAI_API_KEY:
#         return
#     idx = ResumeIndex()
#     idx.build(resumes, OPENAI_API_KEY)
#     st.session_state.resume_index = idx
#     st.session_state.index_built  = True


# # ── Sidebar ────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style="padding:16px 0 8px 0;">
#         <div style="font-size:18px;font-weight:700;color:#e6edf3;">🎯 Resume Screener</div>
#         <div style="font-size:11px;color:#8b949e;margin-top:4px;">3 API Calls · Any Domain · LangChain</div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.divider()

#     # API key status
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
#     if OPENAI_API_KEY:
#         masked = OPENAI_API_KEY[:7] + "••••••••" + OPENAI_API_KEY[-4:]
#         st.markdown(
#             f'<div class="success-box">✓ Loaded from environment<br>'
#             f'<span style="font-family:DM Mono,monospace;font-size:11px;opacity:0.6;">{masked}</span></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="error-box">✗ OPENAI_API_KEY not found<br>'
#             '<span style="font-size:11px;">Add to .env or set as env variable</span></div>',
#             unsafe_allow_html=True,
#         )
#     st.divider()

#     # Top-N slider
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔢 Results to Show</div>', unsafe_allow_html=True)
#     top_n = st.slider("Top N", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")
#     st.markdown(f'<div style="font-size:11px;color:#8b949e;margin-bottom:4px;">Top <b style="color:#e6edf3">{top_n}</b> candidates</div>', unsafe_allow_html=True)
#     st.divider()

#     # Upload
#     api_ok = bool(OPENAI_API_KEY)
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">📤 Upload Resumes</div>', unsafe_allow_html=True)
#     if not api_ok:
#         st.markdown('<div class="warning-box">⚠ Set OPENAI_API_KEY first</div>', unsafe_allow_html=True)

#     uploaded_files = st.file_uploader(
#         "PDF / DOCX / TXT",
#         type=["pdf", "docx", "doc", "txt"],
#         accept_multiple_files=True,
#         label_visibility="collapsed",
#         disabled=not api_ok,
#     )

#     if uploaded_files and api_ok:
#         new_files = parse_errors = 0
#         progress  = st.progress(0)
#         for i, f in enumerate(uploaded_files):
#             fb = f.read()
#             h  = hashlib.sha256(fb).hexdigest()[:16]
#             if h not in st.session_state.parsed_resumes:
#                 parsed = parse_resume(fb, f.name)
#                 if parsed:
#                     st.session_state.parsed_resumes[h] = parsed
#                     new_files += 1
#                 else:
#                     parse_errors += 1
#             progress.progress((i + 1) / len(uploaded_files))
#         progress.empty()

#         if new_files > 0:
#             with st.spinner(f"Embedding {new_files} resume(s)…  [API call #2]"):
#                 rebuild_index()
#             st.markdown(f'<div class="success-box">✓ {new_files} resume(s) indexed</div>', unsafe_allow_html=True)
#         if parse_errors:
#             st.markdown(f'<div class="warning-box">⚠ {parse_errors} file(s) failed to parse</div>', unsafe_allow_html=True)

#     st.divider()

#     # Resume list
#     total = len(st.session_state.parsed_resumes)
#     st.markdown(
#         f'<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
#         f'📁 Indexed <span class="resume-count-badge">{total}</span></div>',
#         unsafe_allow_html=True,
#     )
#     if total == 0:
#         st.markdown('<div style="font-size:12px;color:#484f58;padding:8px 0;">No resumes yet.</div>', unsafe_allow_html=True)
#     else:
#         for r in list(st.session_state.parsed_resumes.values())[:40]:
#             name = r.get("name", r["filename"])
#             exp  = r.get("experience_years", 0)
#             nsk  = len(r.get("skills", []))
#             st.markdown(
#                 f'<div class="resume-item">📄 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{r["filename"]}">'
#                 f'{name}</span><span style="color:#484f58;font-size:10px;">{exp}y·{nsk}sk</span></div>',
#                 unsafe_allow_html=True,
#             )
#         if total > 40:
#             st.markdown(f'<div style="font-size:11px;color:#484f58;">…and {total-40} more</div>', unsafe_allow_html=True)

#     st.divider()
#     if total > 0:
#         if st.button("🗑  Clear All", use_container_width=True):
#             st.session_state.parsed_resumes = {}
#             st.session_state.resume_index   = ResumeIndex()
#             st.session_state.results        = []
#             st.session_state.index_built    = False
#             st.rerun()

#     st.markdown("""
#     <div style="margin-top:auto;padding-top:20px;font-size:11px;color:#484f58;line-height:1.9;">
#     <b style="color:#30363d;">API calls/run:</b> exactly 3<br>
#     <b style="color:#30363d;">Upload calls:</b> 0 (free)<br>
#     <b style="color:#30363d;">Embeddings:</b> text-embedding-3-large<br>
#     <b style="color:#30363d;">LLM:</b> gpt-4o-mini<br>
#     <b style="color:#30363d;">Key:</b> env-only, never in UI
#     </div>
#     """, unsafe_allow_html=True)


# # ── Main area ──────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="app-header">
#     <div>
#         <h1>🎯 AI Resume Screener</h1>
#         <p>3 API calls per run · text-embedding-3-large · BM25 · RRF · LLM Skill Match + Rerank · Any domain</p>
#     </div>
#     <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
#         <span class="badge">3 API CALLS</span>
#         <span class="badge">text-embedding-3-large</span>
#         <span class="badge">BM25 + RRF</span>
#         <span class="badge">ANY DOMAIN</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# total_resumes = len(st.session_state.parsed_resumes)
# c1, c2, c3, c4 = st.columns(4)
# with c1: st.metric("Resumes Indexed", total_resumes)
# with c2:
#     avg = round(sum(len(r.get("skills",[])) for r in st.session_state.parsed_resumes.values()) / max(total_resumes,1), 1)
#     st.metric("Avg Skill Tokens", avg)
# with c3: st.metric("Top Results", top_n)
# with c4: st.metric("API Key", "✓ Set" if OPENAI_API_KEY else "✗ Missing")

# st.markdown("---")

# st.markdown('<div style="font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:8px;">📋 Job Description</div>', unsafe_allow_html=True)
# st.markdown('<div style="font-size:12px;color:#8b949e;margin-bottom:12px;">Paste any JD — tech, sales, HR, marketing, ops, finance. LLM understands it all.</div>', unsafe_allow_html=True)

# jd_text = st.text_area(
#     "JD",
#     height=220,
#     placeholder=(
#         "Tech:     Senior Python Engineer, 4+ yrs, FastAPI, PostgreSQL, Docker, RAG/LLM experience\n\n"
#         "Non-tech: Marketing Manager, 5+ yrs, brand strategy, SEO, Google Analytics, content creation\n\n"
#         "Sales:    Account Executive, SaaS B2B, Salesforce, quota achievement, enterprise deals"
#     ),
#     label_visibility="collapsed",
# )

# can_run = total_resumes > 0 and bool(jd_text.strip()) and api_ok
# col_btn, col_info = st.columns([1, 3])

# with col_btn:
#     analyze_btn = st.button("🔍  Analyze Candidates", use_container_width=True, disabled=not can_run)

# with col_info:
#     if not api_ok:
#         st.markdown('<div class="error-box">✗ Set OPENAI_API_KEY in .env to use this tool.</div>', unsafe_allow_html=True)
#     elif total_resumes == 0:
#         st.markdown('<div class="warning-box">⚠ Upload resumes first.</div>', unsafe_allow_html=True)
#     elif not jd_text.strip():
#         st.markdown('<div class="info-box">ℹ Paste a job description then click Analyze.</div>', unsafe_allow_html=True)
#     else:
#         cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)
#         if cur_hash == st.session_state.last_jd_hash and st.session_state.results:
#             st.markdown('<div class="success-box">✓ Cached — same JD + resumes + top-N → same results.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="info-box">ℹ Ready · {total_resumes} resumes · top {top_n} · 3 API calls</div>', unsafe_allow_html=True)


# # ── Run ────────────────────────────────────────────────────────────────────────
# if analyze_btn and can_run:
#     cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)

#     if cur_hash != st.session_state.last_jd_hash or not st.session_state.results:
#         if not st.session_state.index_built:
#             with st.spinner("Building embedding index… [API call #2]"):
#                 rebuild_index()

#         with st.spinner("Call #1: parsing JD  →  Call #3: matching skills & generating explanations…"):
#             t0      = time.time()
#             results = retrieve_top_n(
#                 st.session_state.resume_index,
#                 jd_text,
#                 top_n=top_n,
#                 api_key=OPENAI_API_KEY,
#             )
#             elapsed = time.time() - t0

#         st.session_state.results      = results
#         st.session_state.last_jd_hash = cur_hash
#         st.session_state.last_elapsed = elapsed
#     else:
#         results = st.session_state.results


# # ── Results ────────────────────────────────────────────────────────────────────
# results = st.session_state.results

# if results:
#     elapsed = st.session_state.get("last_elapsed", 0)
#     st.markdown("---")
#     st.markdown(
#         f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
#         f'<div style="font-size:18px;font-weight:700;color:#e6edf3;">Top {len(results)} Candidates</div>'
#         f'<div style="font-size:12px;color:#8b949e;font-family:DM Mono,monospace;">'
#         f'{elapsed:.2f}s · 3 API calls · text-embedding-3-large + BM25 + RRF + LLM skill match</div>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 Summary Table", expanded=False):
#         rows = []
#         for rank, r in enumerate(results, 1):
#             c, ex = r["candidate"], r["explanation"]
#             rows.append({
#                 "Rank":           rank,
#                 "Name":           c.get("name", c["filename"]),
#                 "Match %":        f"{ex['score_pct']}%",
#                 "Exp (yrs)":      c.get("experience_years", "—"),
#                 "Skills Matched": len(ex["matched_skills"]),
#                 "Required Met":   len(ex["required_matched"]),
#                 "LLM Scored":     "✓" if r.get("ce_score", 0) > 0 else "✗ fallback",
#                 "File":           c["filename"],
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

#     # ── Card CSS ───────────────────────────────────────────────────────────────
#     CARD_CSS = """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
#     *{box-sizing:border-box;margin:0;padding:0}
#     body{background:transparent;font-family:'DM Sans',sans-serif}
#     .rank-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px 24px;margin-bottom:4px;position:relative}
#     .rank-card.top1{border-left:3px solid #f0883e}
#     .rank-card.top2{border-left:3px solid #58a6ff}
#     .rank-card.top3{border-left:3px solid #3fb950}
#     .rank-card.rest{border-left:3px solid #30363d}
#     .rank-number{position:absolute;top:16px;right:20px;font-size:28px;font-weight:700;color:#21262d;font-family:'DM Mono',monospace}
#     .candidate-name{font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:4px}
#     .candidate-meta{font-size:12px;color:#8b949e;margin-bottom:12px;font-family:'DM Mono',monospace}
#     .score-bar-wrap{background:#21262d;border-radius:4px;height:6px;margin:8px 0 4px;overflow:hidden}
#     .score-bar{height:100%;border-radius:4px}
#     .score-label{font-size:24px;font-weight:700;font-family:'DM Mono',monospace}
#     .score-high{color:#3fb950}.score-mid{color:#f0883e}.score-low{color:#f85149}

#     /* FIX: tag spacing — was missing margin, caused tags to appear fused together */
#     .tag{
#         display:inline-block;
#         padding:3px 10px;
#         border-radius:6px;
#         font-size:11px;
#         font-weight:500;
#         margin:2px 4px 2px 0;   /* was: margin:2px 3px 2px 0 — too tight */
#         font-family:'DM Mono',monospace;
#     }
#     .tag-green{background:#1a2e1a;color:#3fb950;border:1px solid #2d5a2d}
#     .tag-red  {background:#2e1a1a;color:#f85149;border:1px solid #5a2d2d}
#     .tag-gray {background:#21262d;color:#8b949e;border:1px solid #30363d}

#     /* FIX: token wrap container — flex-wrap + gap prevents tags from merging */
#     .token-wrap{
#         display:flex;
#         flex-wrap:wrap;
#         gap:4px;         /* consistent gap between tokens */
#         margin-top:4px;
#     }

#     .sec-label{font-size:11px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px}
#     .str-item{font-size:13px;color:#3fb950;padding:2px 0}.str-item::before{content:"✓  "}
#     .gap-item{font-size:13px;color:#f85149;padding:2px 0}.gap-item::before{content:"✗  "}
#     .score-meta{font-size:10px;color:#484f58;font-family:'DM Mono',monospace;margin-top:2px}
#     .fallback-badge{
#         display:inline-block;background:#2e2218;border:1px solid #6e4c1a;
#         color:#e3b341;font-size:10px;padding:1px 7px;border-radius:4px;
#         font-family:'DM Mono',monospace;margin-left:6px;vertical-align:middle;
#     }
#     </style>
#     """

#     html = CARD_CSS
#     for rank, r in enumerate(results, 1):
#         c, ex     = r["candidate"], r["explanation"]
#         score     = r["final_score"]
#         pct       = ex["score_pct"]
#         cc        = score_color(score)
#         bc        = bar_color(score)
#         cl        = card_class(rank)
#         medal     = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
#         exp_disp  = f"{c.get('experience_years', 0)} yrs" if c.get("experience_years") else "—"
#         email     = c.get("email", "")

#         # Score breakdown
#         ce_score  = r.get("ce_score", 0)
#         llm_pct   = round(ce_score * 100)
#         sem_pct   = round(r["semantic_score"] * 100)
#         ep        = round(r["experience_score"] * 100)

#         # FIX: show fallback badge if LLM didn't process this candidate
#         is_fallback  = ce_score == 0.0
#         fallback_tag = '<span class="fallback-badge">heuristic only</span>' if is_fallback else ""

#         # Score meta line
#         if is_fallback:
#             score_meta = f'Embed {sem_pct}% · Exp {ep}% · LLM not applied {fallback_tag}'
#         else:
#             score_meta = f'LLM {llm_pct}% · Embed {sem_pct}% · Exp {ep}%'

#         # Matched skills
#         matched_html = "".join(
#             f'<span class="tag tag-green">{s}</span>'
#             for s in ex["matched_skills"][:12]
#         ) or '<span style="color:#484f58;font-size:12px;">None detected by LLM</span>'

#         miss_html = ""
#         if ex["missing_skills"]:
#             miss_html = (
#                 '<div class="sec-label" style="margin-top:10px;">Missing Skills</div>'
#                 + "".join(f'<span class="tag tag-red">{s}</span>' for s in ex["missing_skills"][:6])
#             )

#         str_html = "".join(
#             f'<div class="str-item">{s}</div>' for s in ex["strengths"][:4]
#         ) or '<div style="color:#484f58;font-size:12px;">—</div>'

#         gap_html = ""
#         if ex["gaps"]:
#             gap_html = (
#                 '<div style="margin-top:6px;">'
#                 + "".join(f'<div class="gap-item">{g}</div>' for g in ex["gaps"][:3])
#                 + "</div>"
#             )

#         # FIX: Resume Tokens — use token-wrap div with gap instead of inline tags
#         # Old: "".join(f'<span class="tag">{s}</span>') — no gap between spans → fused display
#         # New: wrap in .token-wrap div with gap:4px, each tag has margin-right:4px
#         sk_prev      = c.get("skills", [])[:8]
#         all_sk_inner = "".join(f'<span class="tag tag-gray">{s}</span>' for s in sk_prev)
#         extra        = len(c.get("skills", [])) - 8
#         if extra > 0:
#             all_sk_inner += f'<span class="tag tag-gray">+{extra} more</span>'
#         all_sk_html = f'<div class="token-wrap">{all_sk_inner}</div>'

#         html += f"""
#         <div class="rank-card {cl}">
#           <div class="rank-number">{medal}</div>
#           <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
#             <div style="flex:1;min-width:200px;">
#               <div class="candidate-name">{c.get("name", c["filename"])}</div>
#               <div class="candidate-meta">{c["filename"]} &middot; {exp_disp} &middot; {email}</div>
#               <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
#                 <div class="score-label {cc}">{pct}%</div>
#                 <div style="flex:1;">
#                   <div class="score-bar-wrap">
#                     <div class="score-bar" style="width:{pct}%;background:{bc};"></div>
#                   </div>
#                   <div class="score-meta">{score_meta}</div>
#                 </div>
#               </div>
#             </div>
#           </div>
#           <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px;">
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">Matched Skills</div>
#               <div class="token-wrap">{matched_html}</div>
#               {miss_html}
#             </div>
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">Why Selected</div>
#               {str_html}{gap_html}
#             </div>
#           </div>
#           <div style="margin-top:12px;">
#             <div class="sec-label">Resume Tokens</div>
#             {all_sk_html}
#           </div>
#         </div>
#         """

#     components.html(html, height=360 * len(results) + 60, scrolling=False)

#     # ── Export ─────────────────────────────────────────────────────────────────
#     st.markdown("---")
#     export = []
#     for rank, r in enumerate(results, 1):
#         c, ex = r["candidate"], r["explanation"]
#         export.append({
#             "Rank":               rank,
#             "Name":               c.get("name", ""),
#             "File":               c["filename"],
#             "Match_%":            ex["score_pct"],
#             "Experience_Years":   c.get("experience_years", 0),
#             "Email":              c.get("email", ""),
#             "Matched_Skills":     ", ".join(ex["matched_skills"]),
#             "Missing_Skills":     ", ".join(ex["missing_skills"]),
#             "Strengths":          " | ".join(ex["strengths"]),
#             "Gaps":               " | ".join(ex["gaps"]),
#             "LLM_Match_%":        round(r.get("ce_score", 0) * 100, 1),
#             "Embed_Score_%":      round(r["semantic_score"] * 100, 1),
#             "Experience_Score_%": round(r["experience_score"] * 100, 1),
#             "Final_Score_%":      ex["score_pct"],
#             "LLM_Applied":        "Yes" if r.get("ce_score", 0) > 0 else "No (heuristic fallback)",
#         })

#     col1, col2 = st.columns(2)
#     with col1:
#         st.download_button(
#             "📥 Download CSV",
#             data=pd.DataFrame(export).to_csv(index=False),
#             file_name="top_candidates.csv",
#             mime="text/csv",
#             use_container_width=True,
#         )
#     with col2:
#         st.download_button(
#             "📥 Download JSON",
#             data=json.dumps(export, indent=2),
#             file_name="top_candidates.json",
#             mime="application/json",
#             use_container_width=True,
#         )

# elif total_resumes == 0:
#     st.markdown("""
#     <div style="text-align:center;padding:60px 24px;color:#484f58;">
#         <div style="font-size:48px;margin-bottom:16px;">📂</div>
#         <div style="font-size:18px;font-weight:600;color:#30363d;margin-bottom:8px;">No resumes uploaded</div>
#         <div style="font-size:13px;">Upload PDF, DOCX, or TXT files from the sidebar.</div>
#     </div>
#     """, unsafe_allow_html=True)














# """
# AI Resume Screener — Streamlit App (LangChain · 3 API Calls)
# ══════════════════════════════════════════════════════════════════════════════
# API call budget per analysis run:
#   #1  extract_jd_features   — gpt-4o-mini             — JD → skills/exp/summary
#   #2  ResumeIndex.build()   — text-embedding-3-large  — embed all resume chunks
#   #3  rerank_and_explain()  — gpt-4o-mini             — score + explain top-N

# Upload time: 0 API calls (pure text extraction + heuristics).
# API key: loaded ONLY from .env / environment — never entered in UI.
# """

# import os
# import time
# import json
# import hashlib
# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# from dotenv import load_dotenv

# from resume_parser import parse_resume
# from retrieval_engine import ResumeIndex, retrieve_top_n, validate_jd

# load_dotenv()

# # ── Page config ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Resume Screener AI",
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
# [data-testid="collapsedControl"] {
#     display: flex !important; visibility: visible !important; opacity: 1 !important;
# }
# section[data-testid="stSidebar"] {
#     display: flex !important; visibility: visible !important;
#     transform: none !important; min-width: 280px !important;
# }
# section[data-testid="stSidebar"][aria-expanded="false"] {
#     margin-left: 0 !important; transform: none !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── CSS ────────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
# html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
# .stApp { background-color: #0d1117; color: #e6edf3; }
# section[data-testid="stSidebar"] {
#     background-color: #161b22 !important; border-right: 1px solid #21262d;
#     display: flex !important; visibility: visible !important;
#     transform: translateX(0) !important; min-width: 244px !important; max-width: 320px !important;
# }
# section[data-testid="stSidebar"][aria-expanded="false"] {
#     transform: translateX(0) !important; min-width: 244px !important;
# }
# [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
# section[data-testid="stSidebar"] .stMarkdown p,
# section[data-testid="stSidebar"] label { color: #8b949e !important; }
# .app-header {
#     background: linear-gradient(135deg,#1a2332 0%,#0d1117 100%);
#     border:1px solid #21262d; border-radius:12px;
#     padding:24px 32px; margin-bottom:24px;
#     display:flex; align-items:center; gap:16px;
# }
# .app-header h1 { font-size:26px; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.5px; }
# .app-header p  { font-size:13px; color:#8b949e; margin:4px 0 0 0; }
# .badge {
#     background:#1f6feb22; border:1px solid #1f6feb55; color:#58a6ff;
#     font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
#     font-family:'DM Mono',monospace; letter-spacing:0.5px;
# }
# .stButton > button {
#     background:#238636 !important; color:white !important; border:none !important;
#     border-radius:8px !important; font-weight:600 !important; font-size:14px !important;
#     padding:10px 24px !important; width:100%; transition:background 0.2s !important;
# }
# .stButton > button:hover    { background:#2ea043 !important; }
# .stButton > button:disabled { background:#21262d !important; color:#484f58 !important; }
# .stMetric { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }
# .stTextArea textarea {
#     background-color:#161b22 !important; border:1px solid #30363d !important;
#     border-radius:8px !important; color:#e6edf3 !important; font-size:14px !important;
# }
# .stTextArea textarea:focus { border-color:#1f6feb !important; box-shadow:0 0 0 3px #1f6feb22 !important; }
# .info-box    { background:#1a2740; border:1px solid #1f4080; border-radius:8px; padding:12px 16px; font-size:13px; color:#79c0ff; margin:8px 0; }
# .success-box { background:#1a2e1a; border:1px solid #2d5a2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#56d364; margin:8px 0; }
# .warning-box { background:#2e2218; border:1px solid #6e4c1a; border-radius:8px; padding:12px 16px; font-size:13px; color:#e3b341; margin:8px 0; }
# .error-box   { background:#2e1a1a; border:1px solid #5a2d2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#f85149; margin:8px 0; }
# .resume-item {
#     background:#0d1117; border:1px solid #21262d; border-radius:6px;
#     padding:8px 12px; margin:4px 0; font-size:12px; color:#8b949e;
#     display:flex; align-items:center; gap:8px;
# }
# .resume-count-badge {
#     background:#1f6feb33; color:#58a6ff; border-radius:12px;
#     padding:2px 8px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace;
# }
# #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
# ::-webkit-scrollbar { width:6px; }
# ::-webkit-scrollbar-track { background:#0d1117; }
# ::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
# hr { border-color:#21262d !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── Session state ──────────────────────────────────────────────────────────────
# for k, v in {
#     "resume_index":   ResumeIndex(),
#     "parsed_resumes": {},
#     "results":        [],
#     "last_jd_hash":   "",
#     "index_built":    False,
#     "last_elapsed":   0.0,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def make_jd_hash(jd_text: str, resume_hashes: list, top_n: int) -> str:
#     combined = jd_text.strip() + "||" + ",".join(sorted(resume_hashes)) + f"||{top_n}"
#     return hashlib.sha256(combined.encode()).hexdigest()[:12]

# def score_color(s: float) -> str:
#     return "score-high" if s >= 0.70 else ("score-mid" if s >= 0.45 else "score-low")

# def bar_color(s: float) -> str:
#     return "#3fb950" if s >= 0.70 else ("#f0883e" if s >= 0.45 else "#f85149")

# def card_class(rank: int) -> str:
#     return {1: "top1", 2: "top2", 3: "top3"}.get(rank, "rest")

# def rebuild_index():
#     resumes = list(st.session_state.parsed_resumes.values())
#     if not resumes or not OPENAI_API_KEY:
#         return
#     idx = ResumeIndex()
#     idx.build(resumes, OPENAI_API_KEY)
#     st.session_state.resume_index = idx
#     st.session_state.index_built  = True


# # ── Sidebar ────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style="padding:16px 0 8px 0;">
#         <div style="font-size:18px;font-weight:700;color:#e6edf3;">🎯 Resume Screener</div>
#         <div style="font-size:11px;color:#8b949e;margin-top:4px;">3 API Calls · Any Domain · LangChain</div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.divider()

#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
#     if OPENAI_API_KEY:
#         masked = OPENAI_API_KEY[:7] + "••••••••" + OPENAI_API_KEY[-4:]
#         st.markdown(
#             f'<div class="success-box">✓ Loaded from environment<br>'
#             f'<span style="font-family:DM Mono,monospace;font-size:11px;opacity:0.6;">{masked}</span></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="error-box">✗ OPENAI_API_KEY not found<br>'
#             '<span style="font-size:11px;">Add to .env or set as env variable</span></div>',
#             unsafe_allow_html=True,
#         )
#     st.divider()

#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔢 Results to Show</div>', unsafe_allow_html=True)
#     top_n = st.slider("Top N", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")
#     st.markdown(f'<div style="font-size:11px;color:#8b949e;margin-bottom:4px;">Top <b style="color:#e6edf3">{top_n}</b> candidates</div>', unsafe_allow_html=True)
#     st.divider()

#     api_ok = bool(OPENAI_API_KEY)
#     st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">📤 Upload Resumes</div>', unsafe_allow_html=True)
#     if not api_ok:
#         st.markdown('<div class="warning-box">⚠ Set OPENAI_API_KEY first</div>', unsafe_allow_html=True)

#     uploaded_files = st.file_uploader(
#         "PDF / DOCX / TXT",
#         type=["pdf", "docx", "doc", "txt"],
#         accept_multiple_files=True,
#         label_visibility="collapsed",
#         disabled=not api_ok,
#     )

#     if uploaded_files and api_ok:
#         new_files = parse_errors = 0
#         progress  = st.progress(0)
#         for i, f in enumerate(uploaded_files):
#             fb = f.read()
#             h  = hashlib.sha256(fb).hexdigest()[:16]
#             if h not in st.session_state.parsed_resumes:
#                 parsed = parse_resume(fb, f.name)
#                 if parsed:
#                     st.session_state.parsed_resumes[h] = parsed
#                     new_files += 1
#                 else:
#                     parse_errors += 1
#             progress.progress((i + 1) / len(uploaded_files))
#         progress.empty()

#         if new_files > 0:
#             with st.spinner(f"Embedding {new_files} resume(s)…  [API call #2]"):
#                 rebuild_index()
#             st.session_state.last_jd_hash = ""
#             st.session_state.results = []
#             st.markdown(f'<div class="success-box">✓ {new_files} resume(s) indexed</div>', unsafe_allow_html=True)
#         if parse_errors:
#             st.markdown(f'<div class="warning-box">⚠ {parse_errors} file(s) failed to parse</div>', unsafe_allow_html=True)

#     st.divider()

#     total = len(st.session_state.parsed_resumes)
#     st.markdown(
#         f'<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
#         f'📁 Indexed <span class="resume-count-badge">{total}</span></div>',
#         unsafe_allow_html=True,
#     )
#     if total == 0:
#         st.markdown('<div style="font-size:12px;color:#484f58;padding:8px 0;">No resumes yet.</div>', unsafe_allow_html=True)
#     else:
#         for r in list(st.session_state.parsed_resumes.values())[:40]:
#             name = r.get("name", r["filename"])
#             exp  = r.get("experience_years", 0)
#             nsk  = len(r.get("skills", []))
#             st.markdown(
#                 f'<div class="resume-item">📄 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{r["filename"]}">'
#                 f'{name}</span><span style="color:#484f58;font-size:10px;">{exp}y·{nsk}sk</span></div>',
#                 unsafe_allow_html=True,
#             )
#         if total > 40:
#             st.markdown(f'<div style="font-size:11px;color:#484f58;">…and {total-40} more</div>', unsafe_allow_html=True)

#     st.divider()
#     if total > 0:
#         if st.button("🗑  Clear All", use_container_width=True):
#             st.session_state.parsed_resumes = {}
#             st.session_state.resume_index   = ResumeIndex()
#             st.session_state.results        = []
#             st.session_state.index_built    = False
#             st.rerun()

#     st.markdown("""
#     <div style="margin-top:auto;padding-top:20px;font-size:11px;color:#484f58;line-height:1.9;">
#     <b style="color:#30363d;">API calls/run:</b> exactly 3<br>
#     <b style="color:#30363d;">Upload calls:</b> 0 (free)<br>
#     <b style="color:#30363d;">Embeddings:</b> text-embedding-3-large<br>
#     <b style="color:#30363d;">LLM:</b> gpt-4o-mini<br>
#     <b style="color:#30363d;">Key:</b> env-only, never in UI
#     </div>
#     """, unsafe_allow_html=True)


# # ── Main area ──────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="app-header">
#     <div>
#         <h1>🎯 AI Resume Screener</h1>
#         <p>3 API calls per run · text-embedding-3-large · BM25 · RRF · LLM Rerank+Explain · Any domain</p>
#     </div>
#     <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
#         <span class="badge">3 API CALLS</span>
#         <span class="badge">text-embedding-3-large</span>
#         <span class="badge">BM25 + RRF</span>
#         <span class="badge">ANY DOMAIN</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# total_resumes = len(st.session_state.parsed_resumes)
# c1, c2, c3, c4 = st.columns(4)
# with c1: st.metric("Resumes Indexed", total_resumes)
# with c2:
#     avg = round(sum(len(r.get("skills",[])) for r in st.session_state.parsed_resumes.values()) / max(total_resumes,1), 1)
#     st.metric("Avg Skill Tokens", avg)
# with c3: st.metric("Top Results", top_n)
# with c4: st.metric("API Key", "✓ Set" if OPENAI_API_KEY else "✗ Missing")

# st.markdown("---")

# st.markdown('<div style="font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:8px;">📋 Job Description</div>', unsafe_allow_html=True)
# st.markdown('<div style="font-size:12px;color:#8b949e;margin-bottom:12px;">Paste any JD — tech, sales, HR, marketing, ops, finance.</div>', unsafe_allow_html=True)

# jd_text = st.text_area(
#     "JD",
#     height=220,
#     placeholder=(
#         "Tech:     Senior Python Engineer, 4+ yrs, FastAPI, PostgreSQL, Docker, RAG/LLM experience\n\n"
#         "Non-tech: Marketing Manager, 5+ yrs, brand strategy, SEO, Google Analytics, content creation\n\n"
#         "Sales:    Account Executive, SaaS B2B, Salesforce, quota achievement, enterprise deals"
#     ),
#     label_visibility="collapsed",
# )

# can_run = total_resumes > 0 and bool(jd_text.strip()) and api_ok
# col_btn, col_info = st.columns([1, 3])

# with col_btn:
#     analyze_btn = st.button("🔍  Analyze Candidates", use_container_width=True, disabled=not can_run)

# with col_info:
#     if not api_ok:
#         st.markdown('<div class="error-box">✗ Set OPENAI_API_KEY in .env to use this tool.</div>', unsafe_allow_html=True)
#     elif total_resumes == 0:
#         st.markdown('<div class="warning-box">⚠ Upload resumes first.</div>', unsafe_allow_html=True)
#     elif not jd_text.strip():
#         st.markdown('<div class="info-box">ℹ Paste a job description then click Analyze.</div>', unsafe_allow_html=True)
#     else:
#         jd_error = validate_jd(jd_text)
#         if jd_error:
#             st.markdown(f'<div class="warning-box">{jd_error}</div>', unsafe_allow_html=True)
#         else:
#             cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)
#             if cur_hash == st.session_state.last_jd_hash and st.session_state.results:
#                 st.markdown('<div class="success-box">✓ Cached — same JD + resumes + top-N.</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="info-box">ℹ Ready · {total_resumes} resumes · top {top_n} · 3 API calls</div>', unsafe_allow_html=True)

# # ── Run ────────────────────────────────────────────────────────────────────────
# if analyze_btn and can_run:
#     jd_error = validate_jd(jd_text)
#     if jd_error:
#         st.markdown(f'<div class="error-box">{jd_error}</div>', unsafe_allow_html=True)
#         st.stop()
#     cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)

#     if cur_hash != st.session_state.last_jd_hash or not st.session_state.results:
#         if not st.session_state.index_built:
#             with st.spinner("Building embedding index… [API call #2]"):
#                 rebuild_index()

#         with st.spinner("Call #1: parsing JD  →  Call #3: reranking & generating explanations…"):
#             t0      = time.time()
#             results = retrieve_top_n(
#                 st.session_state.resume_index,
#                 jd_text,
#                 top_n=top_n,
#                 api_key=OPENAI_API_KEY,
#             )
#             elapsed = time.time() - t0

#         st.session_state.results      = results
#         st.session_state.last_jd_hash = cur_hash
#         st.session_state.last_elapsed = elapsed
#     else:
#         results = st.session_state.results


# # ── Results ────────────────────────────────────────────────────────────────────
# results = st.session_state.results

# if results:
#     elapsed = st.session_state.get("last_elapsed", 0)
#     st.markdown("---")
#     st.markdown(
#         f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
#         f'<div style="font-size:18px;font-weight:700;color:#e6edf3;">Top {len(results)} Candidates</div>'
#         f'<div style="font-size:12px;color:#8b949e;font-family:DM Mono,monospace;">'
#         f'{elapsed:.2f}s · 3 API calls · text-embedding-3-large + BM25 + RRF + LLM reranker</div>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     with st.expander("📊 Summary Table", expanded=False):
#         rows = []
#         for rank, r in enumerate(results, 1):
#             c, ex = r["candidate"], r["explanation"]
#             rows.append({
#                 "Rank":             rank,
#                 "Name":             c.get("name", c["filename"]),
#                 "Match %":          f"{ex['score_pct']}%",
#                 "Exp (yrs)":        c.get("experience_years", "—"),
#                 "Verified Matched": len(ex["matched_skills"]),
#                 "Claimed Matched":  len(ex.get("claimed_skills", [])),
#                 "Required Met":     len(ex["required_matched"]),
#                 "File":             c["filename"],
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

#     # ── Card CSS ───────────────────────────────────────────────────────────────
#     CARD_CSS = """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
#     *{box-sizing:border-box;margin:0;padding:0}
#     body{background:transparent;font-family:'DM Sans',sans-serif}
#     .rank-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px 24px;margin-bottom:4px;position:relative}
#     .rank-card.top1{border-left:3px solid #f0883e}
#     .rank-card.top2{border-left:3px solid #58a6ff}
#     .rank-card.top3{border-left:3px solid #3fb950}
#     .rank-card.rest{border-left:3px solid #30363d}
#     .rank-number{position:absolute;top:16px;right:20px;font-size:28px;font-weight:700;color:#21262d;font-family:'DM Mono',monospace}
#     .candidate-name{font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:4px}
#     .candidate-meta{font-size:12px;color:#8b949e;margin-bottom:12px;font-family:'DM Mono',monospace}
#     .score-bar-wrap{background:#21262d;border-radius:4px;height:6px;margin:8px 0 4px;overflow:hidden}
#     .score-bar{height:100%;border-radius:4px}
#     .score-label{font-size:24px;font-weight:700;font-family:'DM Mono',monospace}
#     .score-high{color:#3fb950}.score-mid{color:#f0883e}.score-low{color:#f85149}
#     .tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:500;margin:2px 3px 2px 0;font-family:'DM Mono',monospace}
#     .tag-green {background:#1a2e1a;color:#3fb950;border:1px solid #2d5a2d}
#     .tag-amber {background:#2e2218;color:#e3b341;border:1px solid #6e4c1a}
#     .tag-red   {background:#2e1a1a;color:#f85149;border:1px solid #5a2d2d}
#     .tag-gray  {background:#21262d;color:#8b949e;border:1px solid #30363d}
#     .sec-label{font-size:11px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px}
#     .sec-label-amber{font-size:11px;font-weight:600;color:#e3b341;text-transform:uppercase;letter-spacing:1px;margin:10px 0 6px}
#     .str-item{font-size:13px;color:#3fb950;padding:2px 0}.str-item::before{content:"✓  "}
#     .gap-item{font-size:13px;color:#f85149;padding:2px 0}.gap-item::before{content:"✗  "}
#     .score-meta{font-size:10px;color:#484f58;font-family:'DM Mono',monospace;margin-top:2px}
#     .claimed-note{font-size:10px;color:#8b949e;margin-top:4px;font-style:italic}
#     </style>
#     """

#     html = CARD_CSS
#     for rank, r in enumerate(results, 1):
#         c, ex     = r["candidate"], r["explanation"]
#         score     = r["final_score"]
#         pct       = ex["score_pct"]
#         cc        = score_color(score)
#         bc        = bar_color(score)
#         cl        = card_class(rank)
#         medal     = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
#         exp_disp  = f"{c.get('experience_years',0)} yrs" if c.get("experience_years") else "—"
#         email     = c.get("email","")
#         sk_prev   = c.get("skills",[])[:8]
#         sk_pct    = round(r["skill_score"]*100)
#         sem_pct   = round(r["semantic_score"]*100)
#         ep        = round(r["experience_score"]*100)
#         llm_pct   = round(r.get("ce_score",0)*100)

#         # ── Verified matched (green) ──────────────────────────────────────────
#         matched_html = (
#             "".join(f'<span class="tag tag-green">{s}</span>' for s in ex["matched_skills"][:10])
#             or '<span style="color:#484f58;font-size:12px;">None detected</span>'
#         )

#         # ── Claimed-only matched (amber) — listed in skills but not proven ────
#         claimed_skills = ex.get("claimed_skills", [])
#         claimed_html = ""
#         if claimed_skills:
#             claimed_html = (
#                 '<div class="sec-label-amber">⚠ Claimed Only (listed, not proven in projects/experience)</div>'
#                 + "".join(f'<span class="tag tag-amber">{s}</span>' for s in claimed_skills[:8])
#                 + '<div class="claimed-note">These appear only in the skills section — no project/work evidence found.</div>'
#             )

#         # ── Missing skills (red) ──────────────────────────────────────────────
#         miss_html = ""
#         if ex["missing_skills"]:
#             miss_html = (
#                 '<div class="sec-label" style="color:#f85149;margin-top:10px;">Missing Skills</div>'
#                 + "".join(f'<span class="tag tag-red">{s}</span>' for s in ex["missing_skills"][:6])
#             )

#         # ── Why selected (strengths) — sanitised, no contradictions ──────────
#         str_html = (
#             "".join(f'<div class="str-item">{s}</div>' for s in ex["strengths"][:4])
#             or '<div style="color:#484f58;font-size:12px;">—</div>'
#         )

#         # ── Gaps — sanitised, no contradictions ──────────────────────────────
#         gap_html = ""
#         if ex["gaps"]:
#             gap_html = (
#                 '<div style="margin-top:6px;">'
#                 + "".join(f'<div class="gap-item">{g}</div>' for g in ex["gaps"][:3])
#                 + "</div>"
#             )

#         # ── All skill tokens (gray) ───────────────────────────────────────────
#         all_sk_html = "".join(f'<span class="tag tag-gray">{s}</span>' for s in sk_prev)
#         extra = len(c.get("skills",[])) - 8
#         if extra > 0:
#             all_sk_html += f'<span class="tag tag-gray">+{extra} more</span>'

#         html += f"""
#         <div class="rank-card {cl}">
#           <div class="rank-number">{medal}</div>
#           <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
#             <div style="flex:1;min-width:200px;">
#               <div class="candidate-name">{c.get("name", c["filename"])}</div>
#               <div class="candidate-meta">{c["filename"]} &middot; {exp_disp} &middot; {email}</div>
#               <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
#                 <div class="score-label {cc}">{pct}%</div>
#                 <div style="flex:1;">
#                   <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{bc};"></div></div>
#                   <div class="score-meta">Skills {sk_pct}% &middot; Semantic {sem_pct}% &middot; Exp {ep}% &middot; LLM {llm_pct}%</div>
#                 </div>
#               </div>
#             </div>
#           </div>

#           <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px;">
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">✅ Verified Skills (proven in experience/projects)</div>
#               <div>{matched_html}</div>
#               {claimed_html}
#               {miss_html}
#             </div>
#             <div style="flex:1;min-width:200px;">
#               <div class="sec-label">Why Selected</div>
#               {str_html}
#               {gap_html}
#             </div>
#           </div>

#           <div style="margin-top:12px;">
#             <div class="sec-label">All Skill Tokens</div>
#             <div>{all_sk_html}</div>
#           </div>
#         </div>
#         """

#     components.html(html, height=380 * len(results) + 60, scrolling=False)

#     # ── Export ────────────────────────────────────────────────────────────────
#     st.markdown("---")
#     export = []
#     for rank, r in enumerate(results, 1):
#         c, ex = r["candidate"], r["explanation"]
#         export.append({
#             "Rank":                   rank,
#             "Name":                   c.get("name",""),
#             "File":                   c["filename"],
#             "Match_%":                ex["score_pct"],
#             "Experience_Years":       c.get("experience_years",0),
#             "Email":                  c.get("email",""),
#             "Verified_Matched":       ", ".join(ex["matched_skills"]),
#             "Claimed_Only":           ", ".join(ex.get("claimed_skills", [])),
#             "Missing_Skills":         ", ".join(ex["missing_skills"]),
#             "Strengths":              " | ".join(ex["strengths"]),
#             "Gaps":                   " | ".join(ex["gaps"]),
#             "Skill_Score_%":          round(r["skill_score"]*100,1),
#             "Semantic_Score_%":       round(r["semantic_score"]*100,1),
#             "Experience_Score_%":     round(r["experience_score"]*100,1),
#             "LLM_Rerank_%":           round(r.get("ce_score",0)*100,1),
#         })

#     col1, col2 = st.columns(2)
#     with col1:
#         st.download_button("📥 Download CSV",  data=pd.DataFrame(export).to_csv(index=False), file_name="top_candidates.csv",  mime="text/csv",        use_container_width=True)
#     with col2:
#         st.download_button("📥 Download JSON", data=json.dumps(export, indent=2),             file_name="top_candidates.json", mime="application/json", use_container_width=True)

# elif total_resumes == 0:
#     st.markdown("""
#     <div style="text-align:center;padding:60px 24px;color:#484f58;">
#         <div style="font-size:48px;margin-bottom:16px;">📂</div>
#         <div style="font-size:18px;font-weight:600;color:#30363d;margin-bottom:8px;">No resumes uploaded</div>
#         <div style="font-size:13px;">Upload PDF, DOCX, or TXT files from the sidebar.</div>
#     </div>
#     """, unsafe_allow_html=True)



"""
AI Resume Screener — Streamlit App (LangChain · 3 API Calls)
══════════════════════════════════════════════════════════════════════════════
API call budget per analysis run:
  #1  extract_jd_features   — gpt-4o-mini      — JD → skills/exp/summary
  #2  ResumeIndex.build()   — text-embedding-3-large — embed all resume chunks
  #3  rerank_and_explain()  — gpt-4o-mini      — score + explain all top-N

Upload time: 0 API calls (pure text extraction + heuristics).
API key: loaded ONLY from .env / environment — never entered in UI.
"""

import os
import time
import json
import hashlib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv

from resume_parser import parse_resume
from retrieval_engine import ResumeIndex, retrieve_top_n

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar always visible
st.markdown("""
<style>
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] {
    background-color: #161b22 !important; border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #8b949e !important; }
.app-header {
    background: linear-gradient(135deg,#1a2332 0%,#0d1117 100%);
    border:1px solid #21262d; border-radius:12px;
    padding:24px 32px; margin-bottom:24px;
    display:flex; align-items:center; gap:16px;
}
.app-header h1 { font-size:26px; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.5px; }
.app-header p  { font-size:13px; color:#8b949e; margin:4px 0 0 0; }
.badge {
    background:#1f6feb22; border:1px solid #1f6feb55; color:#58a6ff;
    font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
    font-family:'DM Mono',monospace; letter-spacing:0.5px;
}
.stButton > button {
    background:#238636 !important; color:white !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; font-size:14px !important;
    padding:10px 24px !important; width:100%; transition:background 0.2s !important;
}
.stButton > button:hover    { background:#2ea043 !important; }
.stButton > button:disabled { background:#21262d !important; color:#484f58 !important; }
.stMetric { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }
.stTextArea textarea {
    background-color:#161b22 !important; border:1px solid #30363d !important;
    border-radius:8px !important; color:#e6edf3 !important; font-size:14px !important;
}
.stTextArea textarea:focus { border-color:#1f6feb !important; box-shadow:0 0 0 3px #1f6feb22 !important; }
.info-box    { background:#1a2740; border:1px solid #1f4080; border-radius:8px; padding:12px 16px; font-size:13px; color:#79c0ff; margin:8px 0; }
.success-box { background:#1a2e1a; border:1px solid #2d5a2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#56d364; margin:8px 0; }
.warning-box { background:#2e2218; border:1px solid #6e4c1a; border-radius:8px; padding:12px 16px; font-size:13px; color:#e3b341; margin:8px 0; }
.error-box   { background:#2e1a1a; border:1px solid #5a2d2d; border-radius:8px; padding:12px 16px; font-size:13px; color:#f85149; margin:8px 0; }
.resume-item {
    background:#0d1117; border:1px solid #21262d; border-radius:6px;
    padding:8px 12px; margin:4px 0; font-size:12px; color:#8b949e;
    display:flex; align-items:center; gap:8px;
}
.resume-count-badge {
    background:#1f6feb33; color:#58a6ff; border-radius:12px;
    padding:2px 8px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace;
}
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
hr { border-color:#21262d !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "resume_index":   ResumeIndex(),
    "parsed_resumes": {},
    "results":        [],
    "last_jd_hash":   "",
    "index_built":    False,
    "last_elapsed":   0.0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API key (env only) ────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_jd_hash(jd_text: str, resume_hashes: list, top_n: int) -> str:
    combined = jd_text.strip() + "||" + ",".join(sorted(resume_hashes)) + f"||{top_n}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]

def score_color(s: float) -> str:
    return "score-high" if s >= 0.70 else ("score-mid" if s >= 0.45 else "score-low")

def bar_color(s: float) -> str:
    return "#3fb950" if s >= 0.70 else ("#f0883e" if s >= 0.45 else "#f85149")

def card_class(rank: int) -> str:
    return {1: "top1", 2: "top2", 3: "top3"}.get(rank, "rest")

def rebuild_index():
    resumes = list(st.session_state.parsed_resumes.values())
    if not resumes or not OPENAI_API_KEY:
        return
    idx = ResumeIndex()
    idx.build(resumes, OPENAI_API_KEY)   # ← API Call #2
    st.session_state.resume_index = idx
    st.session_state.index_built  = True


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
        <div style="font-size:18px;font-weight:700;color:#e6edf3;">🎯 Resume Screener</div>
        <div style="font-size:11px;color:#8b949e;margin-top:4px;">3 API Calls · Any Domain · LangChain</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # API key status
    st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
    if OPENAI_API_KEY:
        masked = OPENAI_API_KEY[:7] + "••••••••" + OPENAI_API_KEY[-4:]
        st.markdown(
            f'<div class="success-box">✓ Loaded from environment<br>'
            f'<span style="font-family:DM Mono,monospace;font-size:11px;opacity:0.6;">{masked}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="error-box">✗ OPENAI_API_KEY not found<br>'
            '<span style="font-size:11px;">Add to .env or set as env variable</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()

    # Top-N slider
    st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔢 Results to Show</div>', unsafe_allow_html=True)
    top_n = st.slider("Top N", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:11px;color:#8b949e;margin-bottom:4px;">Top <b style="color:#e6edf3">{top_n}</b> candidates</div>', unsafe_allow_html=True)
    st.divider()

    # Upload
    api_ok = bool(OPENAI_API_KEY)
    st.markdown('<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">📤 Upload Resumes</div>', unsafe_allow_html=True)
    if not api_ok:
        st.markdown('<div class="warning-box">⚠ Set OPENAI_API_KEY first</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not api_ok,
    )

    if uploaded_files and api_ok:
        new_files = parse_errors = 0
        progress  = st.progress(0)
        for i, f in enumerate(uploaded_files):
            fb = f.read()
            h  = hashlib.sha256(fb).hexdigest()[:16]
            if h not in st.session_state.parsed_resumes:
                parsed = parse_resume(fb, f.name)   # 0 API calls
                if parsed:
                    st.session_state.parsed_resumes[h] = parsed
                    new_files += 1
                else:
                    parse_errors += 1
            progress.progress((i + 1) / len(uploaded_files))
        progress.empty()

        if new_files > 0:
            with st.spinner(f"Embedding {new_files} resume(s)…  [API call #2]"):
                rebuild_index()
            st.session_state.last_jd_hash = ""
            st.session_state.results = []
            st.markdown(f'<div class="success-box">✓ {new_files} resume(s) indexed</div>', unsafe_allow_html=True)
        if parse_errors:
            st.markdown(f'<div class="warning-box">⚠ {parse_errors} file(s) failed to parse</div>', unsafe_allow_html=True)

    st.divider()

    # Resume list
    total = len(st.session_state.parsed_resumes)
    st.markdown(
        f'<div style="font-size:12px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
        f'📁 Indexed <span class="resume-count-badge">{total}</span></div>',
        unsafe_allow_html=True,
    )
    if total == 0:
        st.markdown('<div style="font-size:12px;color:#484f58;padding:8px 0;">No resumes yet.</div>', unsafe_allow_html=True)
    else:
        for r in list(st.session_state.parsed_resumes.values())[:40]:
            name = r.get("name", r["filename"])
            exp  = r.get("experience_years", 0)
            nsk  = len(r.get("skills", []))
            st.markdown(
                f'<div class="resume-item">📄 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{r["filename"]}">'
                f'{name}</span><span style="color:#484f58;font-size:10px;">{exp}y·{nsk}sk</span></div>',
                unsafe_allow_html=True,
            )
        if total > 40:
            st.markdown(f'<div style="font-size:11px;color:#484f58;">…and {total-40} more</div>', unsafe_allow_html=True)

    st.divider()
    if total > 0:
        if st.button("🗑  Clear All", use_container_width=True):
            st.session_state.parsed_resumes = {}
            st.session_state.resume_index   = ResumeIndex()
            st.session_state.results        = []
            st.session_state.index_built    = False
            st.rerun()

    st.markdown("""
    <div style="margin-top:auto;padding-top:20px;font-size:11px;color:#484f58;line-height:1.9;">
    <b style="color:#30363d;">API calls/run:</b> exactly 3<br>
    <b style="color:#30363d;">Upload calls:</b> 0 (free)<br>
    <b style="color:#30363d;">Embeddings:</b> text-embedding-3-large<br>
    <b style="color:#30363d;">LLM:</b> gpt-4o-mini<br>
    <b style="color:#30363d;">Key:</b> env-only, never in UI
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>🎯 AI Resume Screener</h1>
        <p>3 API calls per run · text-embedding-3-large · BM25 · RRF · LLM Rerank+Explain · Any domain</p>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <span class="badge">3 API CALLS</span>
        <span class="badge">text-embedding-3-large</span>
        <span class="badge">BM25 + RRF</span>
        <span class="badge">ANY DOMAIN</span>
    </div>
</div>
""", unsafe_allow_html=True)

total_resumes = len(st.session_state.parsed_resumes)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Resumes Indexed", total_resumes)
with c2:
    avg = round(sum(len(r.get("skills",[])) for r in st.session_state.parsed_resumes.values()) / max(total_resumes,1), 1)
    st.metric("Avg Skill Tokens", avg)
with c3: st.metric("Top Results", top_n)
with c4: st.metric("API Key", "✓ Set" if OPENAI_API_KEY else "✗ Missing")

st.markdown("---")

st.markdown('<div style="font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:8px;">📋 Job Description</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:12px;color:#8b949e;margin-bottom:12px;">Paste any JD — tech, sales, HR, marketing, ops, finance. LLM understands it all.</div>', unsafe_allow_html=True)

jd_text = st.text_area(
    "JD",
    height=220,
    placeholder=(
        "Tech:     Senior Python Engineer, 4+ yrs, FastAPI, PostgreSQL, Docker, RAG/LLM experience\n\n"
        "Non-tech: Marketing Manager, 5+ yrs, brand strategy, SEO, Google Analytics, content creation\n\n"
        "Sales:    Account Executive, SaaS B2B, Salesforce, quota achievement, enterprise deals"
    ),
    label_visibility="collapsed",
)

can_run = total_resumes > 0 and bool(jd_text.strip()) and api_ok
col_btn, col_info = st.columns([1, 3])

with col_btn:
    analyze_btn = st.button("🔍  Analyze Candidates", use_container_width=True, disabled=not can_run)

with col_info:
    if not api_ok:
        st.markdown('<div class="error-box">✗ Set OPENAI_API_KEY in .env to use this tool.</div>', unsafe_allow_html=True)
    elif total_resumes == 0:
        st.markdown('<div class="warning-box">⚠ Upload resumes first.</div>', unsafe_allow_html=True)
    elif not jd_text.strip():
        st.markdown('<div class="info-box">ℹ Paste a job description then click Analyze.</div>', unsafe_allow_html=True)
    else:
        cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)
        if cur_hash == st.session_state.last_jd_hash and st.session_state.results:
            st.markdown('<div class="success-box">✓ Cached — same JD + resumes + top-N → same results.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">ℹ Ready · {total_resumes} resumes · top {top_n} · 3 API calls</div>', unsafe_allow_html=True)


# ── Run ────────────────────────────────────────────────────────────────────────
if analyze_btn and can_run:
    cur_hash = make_jd_hash(jd_text, list(st.session_state.parsed_resumes.keys()), top_n)

    if cur_hash != st.session_state.last_jd_hash or not st.session_state.results:
        if not st.session_state.index_built:
            with st.spinner("Building embedding index… [API call #2]"):
                rebuild_index()

        with st.spinner("Call #1: parsing JD  →  Call #3: reranking & generating explanations…"):
            t0      = time.time()
            results = retrieve_top_n(
                st.session_state.resume_index,
                jd_text,
                top_n=top_n,
                api_key=OPENAI_API_KEY,
            )
            elapsed = time.time() - t0

        st.session_state.results      = results
        st.session_state.last_jd_hash = cur_hash
        st.session_state.last_elapsed = elapsed
    else:
        results = st.session_state.results


# ── Results ────────────────────────────────────────────────────────────────────
results = st.session_state.results

if results:
    elapsed = st.session_state.get("last_elapsed", 0)
    st.markdown("---")
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
        f'<div style="font-size:18px;font-weight:700;color:#e6edf3;">Top {len(results)} Candidates</div>'
        f'<div style="font-size:12px;color:#8b949e;font-family:DM Mono,monospace;">'
        f'{elapsed:.2f}s · 3 API calls · text-embedding-3-large + BM25 + RRF + LLM reranker</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📊 Summary Table", expanded=False):
        rows = []
        for rank, r in enumerate(results, 1):
            c, ex = r["candidate"], r["explanation"]
            rows.append({
                "Rank":           rank,
                "Name":           c.get("name", c["filename"]),
                "Match %":        f"{ex['score_pct']}%",
                "Exp (yrs)":      c.get("experience_years", "—"),
                "Skills Matched": len(ex["matched_skills"]),
                "Required Met":   len(ex["required_matched"]),
                "File":           c["filename"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Cards
    CARD_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:transparent;font-family:'DM Sans',sans-serif}
    .rank-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px 24px;margin-bottom:4px;position:relative}
    .rank-card.top1{border-left:3px solid #f0883e}
    .rank-card.top2{border-left:3px solid #58a6ff}
    .rank-card.top3{border-left:3px solid #3fb950}
    .rank-card.rest{border-left:3px solid #30363d}
    .rank-number{position:absolute;top:16px;right:20px;font-size:28px;font-weight:700;color:#21262d;font-family:'DM Mono',monospace}
    .candidate-name{font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:4px}
    .candidate-meta{font-size:12px;color:#8b949e;margin-bottom:12px;font-family:'DM Mono',monospace}
    .score-bar-wrap{background:#21262d;border-radius:4px;height:6px;margin:8px 0 4px;overflow:hidden}
    .score-bar{height:100%;border-radius:4px}
    .score-label{font-size:24px;font-weight:700;font-family:'DM Mono',monospace}
    .score-high{color:#3fb950}.score-mid{color:#f0883e}.score-low{color:#f85149}
    .tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:500;margin:2px 3px 2px 0;font-family:'DM Mono',monospace}
    .tag-green{background:#1a2e1a;color:#3fb950;border:1px solid #2d5a2d}
    .tag-red  {background:#2e1a1a;color:#f85149;border:1px solid #5a2d2d}
    .tag-gray {background:#21262d;color:#8b949e;border:1px solid #30363d}
    .sec-label{font-size:11px;font-weight:600;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px}
    .str-item{font-size:13px;color:#3fb950;padding:2px 0}.str-item::before{content:"✓  "}
    .gap-item{font-size:13px;color:#f85149;padding:2px 0}.gap-item::before{content:"✗  "}
    .score-meta{font-size:10px;color:#484f58;font-family:'DM Mono',monospace;margin-top:2px}
    </style>
    """

    html = CARD_CSS
    for rank, r in enumerate(results, 1):
        c, ex     = r["candidate"], r["explanation"]
        score     = r["final_score"]
        pct       = ex["score_pct"]
        cc        = score_color(score)
        bc        = bar_color(score)
        cl        = card_class(rank)
        medal     = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
        exp_disp  = f"{c.get('experience_years',0)} yrs" if c.get("experience_years") else "—"
        email     = c.get("email","")
        sk_prev   = c.get("skills",[])[:8]
        sk_pct    = round(r["skill_score"]*100)
        sem_pct   = round(r["semantic_score"]*100)
        ep        = round(r["experience_score"]*100)
        llm_pct   = round(r.get("ce_score",0)*100)

        matched_html = "".join(f'<span class="tag tag-green">{s}</span>' for s in ex["matched_skills"][:10]) \
                    or '<span style="color:#484f58;font-size:12px;">None detected</span>'
        miss_html    = (
            '<div class="sec-label" style="margin-top:10px;">Missing Skills</div>'
            + "".join(f'<span class="tag tag-red">{s}</span>' for s in ex["missing_skills"][:6])
        ) if ex["missing_skills"] else ""
        str_html     = "".join(f'<div class="str-item">{s}</div>' for s in ex["strengths"][:4]) \
                    or '<div style="color:#484f58;font-size:12px;">—</div>'
        gap_html     = '<div style="margin-top:6px;">' + "".join(f'<div class="gap-item">{g}</div>' for g in ex["gaps"][:3]) + "</div>" \
                    if ex["gaps"] else ""
        all_sk_html  = "".join(f'<span class="tag tag-gray">{s}</span>' for s in sk_prev)
        extra        = len(c.get("skills",[])) - 8
        if extra > 0:
            all_sk_html += f'<span class="tag tag-gray">+{extra} more</span>'

        html += f"""
        <div class="rank-card {cl}">
          <div class="rank-number">{medal}</div>
          <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
              <div class="candidate-name">{c.get("name", c["filename"])}</div>
              <div class="candidate-meta">{c["filename"]} &middot; {exp_disp} &middot; {email}</div>
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <div class="score-label {cc}">{pct}%</div>
                <div style="flex:1;">
                  <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{bc};"></div></div>
                  <div class="score-meta">Skills {sk_pct}% &middot; Semantic {sem_pct}% &middot; Exp {ep}% &middot; LLM {llm_pct}%</div>
                </div>
              </div>
            </div>
          </div>
          <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px;">
            <div style="flex:1;min-width:200px;">
              <div class="sec-label">Matched Skills</div><div>{matched_html}</div>{miss_html}
            </div>
            <div style="flex:1;min-width:200px;">
              <div class="sec-label">Why Selected</div>{str_html}{gap_html}
            </div>
          </div>
          <div style="margin-top:12px;">
            <div class="sec-label">All Skill Tokens</div><div>{all_sk_html}</div>
          </div>
        </div>
        """

    components.html(html, height=340 * len(results) + 60, scrolling=False)

    # Export
    st.markdown("---")
    export = []
    for rank, r in enumerate(results, 1):
        c, ex = r["candidate"], r["explanation"]
        export.append({
            "Rank":               rank,
            "Name":               c.get("name",""),
            "File":               c["filename"],
            "Match_%":            ex["score_pct"],
            "Experience_Years":   c.get("experience_years",0),
            "Email":              c.get("email",""),
            "Matched_Skills":     ", ".join(ex["matched_skills"]),
            "Missing_Skills":     ", ".join(ex["missing_skills"]),
            "Strengths":          " | ".join(ex["strengths"]),
            "Gaps":               " | ".join(ex["gaps"]),
            "Skill_Score_%":      round(r["skill_score"]*100,1),
            "Semantic_Score_%":   round(r["semantic_score"]*100,1),
            "Experience_Score_%": round(r["experience_score"]*100,1),
            "LLM_Rerank_%":       round(r.get("ce_score",0)*100,1),
        })

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download CSV",  data=pd.DataFrame(export).to_csv(index=False), file_name="top_candidates.csv",  mime="text/csv",        use_container_width=True)
    with col2:
        st.download_button("📥 Download JSON", data=json.dumps(export, indent=2),             file_name="top_candidates.json", mime="application/json", use_container_width=True)

elif total_resumes == 0:
    st.markdown("""
    <div style="text-align:center;padding:60px 24px;color:#484f58;">
        <div style="font-size:48px;margin-bottom:16px;">📂</div>
        <div style="font-size:18px;font-weight:600;color:#30363d;margin-bottom:8px;">No resumes uploaded</div>
        <div style="font-size:13px;">Upload PDF, DOCX, or TXT files from the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)