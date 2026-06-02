"""
Generate sample resumes for testing + run a quick smoke test of the pipeline.
"""
import sys
sys.path.insert(0, "/home/claude/resume_screener")

from resume_parser import parse_resume
from retrieval_engine import ResumeIndex, retrieve_top_n

SAMPLE_RESUMES = [
    ("rahul_sharma.txt", """
Rahul Sharma
rahul.sharma@email.com | +91-9876543210 | Pune, India

SUMMARY
Senior Python Engineer with 5 years of experience building scalable backend systems and AI applications.

SKILLS
Python, FastAPI, LangChain, LangGraph, PostgreSQL, Redis, Docker, Kubernetes, AWS, FAISS, 
OpenAI, RAG, LLM, REST API, Git, GitHub, Microservices, Elasticsearch

EXPERIENCE
Senior Backend Engineer — TechCorp India (2021–Present)
- Built RAG-based chatbot system using LangChain + FAISS + OpenAI GPT-4
- Designed FastAPI microservices deployed on AWS ECS
- PostgreSQL + Redis caching for 10M+ users

Backend Engineer — StartupXYZ (2019–2021)
- Developed REST APIs using Python Flask
- Managed PostgreSQL databases and Redis caches
- CI/CD with GitHub Actions and Docker

PROJECTS
- RAG Chatbot: Built production RAG system with FAISS vector store and GPT-4 for internal knowledge base
- AI Resume Screener: LangChain + embeddings for semantic search over resumes

EDUCATION
B.Tech Computer Science — Pune University (2019)

CERTIFICATIONS
AWS Certified Developer
"""),

    ("priya_patel.txt", """
Priya Patel
priya.patel@email.com | Mumbai, India

SUMMARY
Machine Learning Engineer with 4 years of experience in NLP, deep learning, and LLM applications.

SKILLS
Python, PyTorch, TensorFlow, Hugging Face, Transformers, BERT, LLM, Fine-tuning, 
FastAPI, scikit-learn, pandas, numpy, Docker, Azure, SQL, Git

EXPERIENCE
ML Engineer — DataDriven Ltd (2022–Present)
- Fine-tuned BERT models for classification tasks
- Built NLP pipelines for named entity recognition
- Deployed models using FastAPI on Azure

Junior ML Engineer — AI Labs (2020–2022)
- Implemented text classification using scikit-learn
- Data preprocessing with pandas and numpy
- Experimented with PyTorch deep learning models

PROJECTS
- Sentiment Analysis API: FastAPI + DistilBERT deployed on Azure
- Document Classifier: Fine-tuned BERT for legal document classification

EDUCATION
M.Tech AI/ML — IIT Bombay (2020)
"""),

    ("arjun_mehta.txt", """
Arjun Mehta
arjun.mehta@email.com | Bangalore, India

SUMMARY
Full-Stack Developer with 3 years of experience in React, Node.js, and Python backends.

SKILLS
JavaScript, TypeScript, React, NextJS, Node.js, Express, Python, Flask, MongoDB, 
PostgreSQL, Docker, AWS S3, Lambda, Git, REST API, GraphQL

EXPERIENCE
Full-Stack Developer — WebCo (2021–Present)
- Built React dashboards with TypeScript
- Node.js + Express REST APIs
- AWS Lambda and S3 for serverless backend

Junior Developer — AppStudio (2020–2021)  
- React frontend components
- Python Flask backend APIs
- MongoDB database management

EDUCATION
B.Tech Information Technology — VIT University (2020)
"""),

    ("sneha_joshi.txt", """
Sneha Joshi
sneha.joshi@email.com | Hyderabad, India

SUMMARY
DevOps Engineer with 6 years experience in cloud infrastructure, Kubernetes, and CI/CD pipelines.

SKILLS
AWS, Azure, GCP, Kubernetes, Docker, Terraform, Ansible, Jenkins, GitHub Actions,
Python, Bash, Linux, Helm, CI/CD, Redis, PostgreSQL, EKS, ECS, S3

EXPERIENCE
Senior DevOps Engineer — CloudFirst (2021–Present)
- Managed Kubernetes clusters on AWS EKS for 50+ services
- Built Terraform infrastructure-as-code for multi-region deployments
- Jenkins + GitHub Actions CI/CD pipelines

DevOps Engineer — Infosys (2018–2021)
- Docker containerization of legacy applications
- AWS infrastructure management (EC2, RDS, S3)
- Ansible automation for configuration management

EDUCATION
B.E. Computer Engineering — BITS Pilani (2018)

CERTIFICATIONS
AWS Solutions Architect Professional
CKA (Certified Kubernetes Administrator)
"""),

    ("vikram_nair.txt", """
Vikram Nair
vikram.nair@email.com | Chennai, India

SUMMARY
Data Engineer with 4 years experience building data pipelines, ETL systems, and analytics platforms.

SKILLS
Python, Spark, Kafka, Airflow, dbt, PostgreSQL, MySQL, AWS Glue, Redshift, 
S3, pandas, SQL, Docker, Git, Tableau

EXPERIENCE
Senior Data Engineer — DataPipeline Co (2022–Present)
- Built real-time streaming pipelines using Kafka + Spark
- Apache Airflow DAGs for ETL orchestration
- dbt transformations for analytics warehouse

Data Engineer — Analytics Corp (2020–2022)
- Python ETL scripts with pandas
- AWS Glue and Redshift for data warehouse
- MySQL to PostgreSQL migration

EDUCATION
B.Tech CSE — Anna University (2020)
"""),

    ("kavya_reddy.txt", """
Kavya Reddy
kavya.reddy@email.com | Pune, India

SUMMARY
AI/LLM Engineer with 3 years building RAG systems, vector databases, and generative AI applications.

SKILLS
Python, LangChain, LangGraph, OpenAI, Claude, Gemini, RAG, FAISS, Pinecone, Qdrant,
FastAPI, PostgreSQL, Docker, AWS, Redis, Elasticsearch, Embedding, Vector, LLM, NLP,
Hugging Face, Sentence Transformers, Git

EXPERIENCE
AI Engineer — GenAI Startup (2022–Present)
- Built production RAG pipelines with LangChain + Pinecone for enterprise clients
- Designed multi-agent systems using LangGraph
- Fine-tuned embedding models on domain-specific data using Hugging Face

Junior AI Developer — TechLabs (2021–2022)
- Implemented semantic search using FAISS + Sentence Transformers
- FastAPI endpoints for LLM services

PROJECTS
- Enterprise RAG System: LangChain + Pinecone + GPT-4 with 92% accuracy
- Multi-Agent Workflow: LangGraph orchestration for complex document analysis

EDUCATION
M.Tech Computer Science — IIIT Hyderabad (2021)
"""),

    ("amit_sharma.txt", """
Amit Sharma
amit.sharma@email.com | Delhi, India

SUMMARY
Backend Java developer with 7 years of experience in Spring Boot microservices and enterprise systems.

SKILLS
Java, Spring Boot, Spring Security, Kotlin, Maven, Gradle, PostgreSQL, MongoDB,
Redis, Kafka, Docker, Kubernetes, AWS, Git, REST API, Microservices, JUnit

EXPERIENCE
Senior Java Developer — Enterprise Solutions (2019–Present)
- Spring Boot microservices architecture for banking applications
- Kafka event-driven architecture with 1M+ events/day
- PostgreSQL optimization and Redis caching strategies

Java Developer — IT Services Corp (2017–2019)
- Spring MVC applications for insurance domain
- Oracle database queries and stored procedures

EDUCATION
B.Tech CS — Delhi University (2017)

CERTIFICATIONS
Oracle Java SE 11 Developer
Spring Professional Certification
"""),

    ("ananya_kumar.txt", """
Ananya Kumar
ananya.kumar@email.com | Bengaluru, India

SUMMARY
Python developer with 2 years of experience. Worked on web scraping, automation and REST APIs.

SKILLS
Python, Flask, Django, BeautifulSoup, Selenium, REST API, MySQL, Git, Linux, Bash

EXPERIENCE
Python Developer — WebScraper Inc (2022–Present)
- Built web scrapers using BeautifulSoup and Selenium
- Flask REST APIs for internal tools
- MySQL database design

EDUCATION
B.Tech IT — Bangalore University (2022)
"""),

    ("rohan_desai.txt", """
Rohan Desai
rohan.desai@email.com | Pune, India

SUMMARY
Experienced Python + GenAI engineer specializing in LLM applications, RAG systems and AI backend development. 5+ years of industry experience.

SKILLS
Python, FastAPI, Django, LangChain, OpenAI, RAG, FAISS, Chroma, PostgreSQL, 
Redis, Docker, Kubernetes, AWS, Azure, Elasticsearch, BM25, Sentence Transformers,
Hugging Face, Vector, Embedding, NLP, Git, Microservices, REST API, GraphQL

EXPERIENCE
Senior AI Engineer — AI Product Co (2021–Present)
- Led RAG system development from scratch: chunking, embedding, retrieval, reranking
- FastAPI backend with async endpoints serving 100K+ daily requests
- Elasticsearch + FAISS hybrid retrieval pipeline
- Deployed on Kubernetes with auto-scaling

Backend Engineer — SoftwareCo (2019–2021)
- Python FastAPI microservices
- PostgreSQL + Redis architecture
- Docker containerization

PROJECTS
- Resume Screener AI: Built AI-powered resume matching using semantic search + BM25
- Legal RAG: Document Q&A system for 500K+ legal documents using LangChain + Chroma
- API Gateway: FastAPI-based API gateway with JWT, rate limiting, caching

EDUCATION
B.Tech CS — Pune Institute of Computer Technology (2019)

CERTIFICATIONS
AWS Certified Machine Learning Specialty
"""),

    ("meera_iyer.txt", """
Meera Iyer
meera.iyer@email.com | Coimbatore, India

SUMMARY
Data Scientist with 3.5 years of experience in statistical modeling, ML, and business analytics.

SKILLS
Python, R, scikit-learn, pandas, numpy, matplotlib, seaborn, SQL, PostgreSQL,
Tableau, Power BI, TensorFlow, XGBoost, LightGBM, Jupyter, Git, Docker

EXPERIENCE
Data Scientist — Analytics Agency (2021–Present)
- Built XGBoost models for customer churn prediction (87% accuracy)
- Statistical analysis and A/B testing for product teams
- Tableau dashboards for business stakeholders

Junior Data Scientist — Research Labs (2020–2021)
- R statistical modeling for clinical data
- Python scikit-learn ML pipelines
- SQL queries for data extraction

EDUCATION
M.Sc Statistics — IIT Madras (2020)
"""),
]


def run_test():
    print("=" * 60)
    print("Resume Screener — System Test")
    print("=" * 60)

    # Parse
    parsed = []
    for fname, text in SAMPLE_RESUMES:
        file_bytes = text.encode("utf-8")
        p = parse_resume(file_bytes, fname)
        parsed.append(p)
        print(f"✓ Parsed: {p['name']} | Skills: {len(p['skills'])} | Exp: {p['experience_years']}y")

    print(f"\n{len(parsed)} resumes parsed.")

    # Index
    print("\nBuilding index...")
    idx = ResumeIndex()
    idx.build(parsed)
    print(f"Index ready. {len(idx.chunk_meta)} chunks indexed.")

    # Query
    jd = """
    We are looking for a Senior Python + GenAI Engineer with 4+ years of experience.

    Required:
    - Python, FastAPI
    - RAG systems, LLM experience
    - LangChain or LangGraph
    - Vector databases (FAISS, Pinecone, Qdrant)
    - PostgreSQL, Redis

    Nice to have:
    - Kubernetes, Docker
    - Elasticsearch
    - Hugging Face, Sentence Transformers
    
    3+ years experience required.
    """

    print("\nRunning retrieval (Run 1)...")
    results1 = retrieve_top_n(idx, jd, top_n=5)

    print("\nRunning retrieval (Run 2 — must be identical)...")
    results2 = retrieve_top_n(idx, jd, top_n=5)

    print("\n🏆 Top 5 Results:")
    for rank, (r1, r2) in enumerate(zip(results1, results2), 1):
        match = r1["candidate"]["filename"] == r2["candidate"]["filename"]
        print(f"  {rank}. {r1['candidate']['name']:<20} {r1['explanation']['score_pct']}%  {'✓ DETERMINISTIC' if match else '✗ MISMATCH!'}")

    # Verify determinism
    names1 = [r["candidate"]["filename"] for r in results1]
    names2 = [r["candidate"]["filename"] for r in results2]
    assert names1 == names2, "DETERMINISM FAILED!"
    print("\n✅ DETERMINISM TEST PASSED — Identical inputs produce identical outputs.")
    print("✅ System is ready.")


if __name__ == "__main__":
    run_test()