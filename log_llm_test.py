#!/usr/bin/env python3
"""
enhanced_sequential_agent.py

– 더미 DB에서 트레이스 읽기
– 중복 제거
– IsolationForest ML 필터로 고위험 후보 선정
– Chroma 벡터 스토어로 과거 로그에서 유사 사례 검색 (RAG)
– Gemma 2B 모델로 한 문장 요약
– GPT-4o-mini로 notify/ignore 결정
– Slack Webhook으로 결과 전송
"""

""" 
requirements.txt

pip install \
  numpy \
  python-dotenv \
  scikit-learn \
  sentence-transformers \
  chromadb \
  langchain-ollama \
  langchain-openai \
  requests

"""


import os, json, sqlite3, requests, re
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# ml / embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# LLM
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI


# 환경 변수 & 설정
load_dotenv()
DB_PATH = "logs.db"
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemma 2B 양자화 모델 (로컬 Ollama 서버에서 실행 중이어야 함 / 따로 터미널 띄워놓고 실행하기)
gemma = Ollama(model="gemma:2b-instruct-q4_0", temperature=0)
# GPT-4o-mini (OpenAI API)
gpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)


# Chroma 벡터 스토어 초기화
def init_rag_store():
    # chromadb 버전별로 encode_kwargs가 없을 수도 있으니
    # 우선 model_name만 넘기기
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()
    return client.get_or_create_collection("logs", embedding_function=ef)


# 더미 db 초기화
def init_dummy_db():
    if os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE traces (
                    id   INTEGER PRIMARY KEY,
                    ts   TEXT,  
                    host TEXT,
                    msg  TEXT
                )"""
    )
    demo = [
        ("2025-06-12 10:01:00", "srv1", "sshd: Failed password for root from 1.2.3.4"),
        ("2025-06-12 10:05:03", "srv2", "nginx: GET /etc/passwd HTTP/1.1"),
        ("2025-06-12 10:07:10", "srv3", "user bob executed sudo su"),
        ("2025-06-11 09:00:00", "srv1", "sshd: Accepted password for alice"),
        ("2025-06-11 09:05:00", "srv2", "nginx: GET /index.html HTTP/1.1"),
    ]
    cursor.executemany("INSERT INTO traces(ts,host,msg) VALUES (?,?,?)", demo)
    conn.commit()
    conn.close()


def fetch_traces(limit=10):  # 최대 10개의 트레이스
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def dedup(traces):
    seen, out = set(), []
    for t in traces:
        sig = hash((t["host"], t["msg"]))
        if sig not in seen:
            seen.add(sig)
            out.append(t)
    return out


# 전처리 + ML 필터링 (IsolationForest)
def preprocess(msg):
    # 숫자/IP 주소를 <*>로 치환
    return re.sub(r"\b[\d\.]+\b", "<*>", msg)


def filter_anomalies(traces, contamination=0.3):
    # 메시지 전처리
    msgs = [preprocess(t["msg"]) for t in traces]
    # TF-IDF 벡터화
    vect = TfidfVectorizer()
    X = vect.fit_transform(msgs).toarray()
    # IsolationForest 스코어
    clf = IsolationForest(contamination=contamination, random_state=0)
    clf.fit(X)
    scores = -clf.decision_function(X)
    thr = np.percentile(scores, 100 * (1 - contamination))
    # 임계치 이상인 것만 리턴
    return [traces[i] for i, s in enumerate(scores) if s >= thr]


# chroma (일종의 벡터 스토어) 로 과거 로그 검색하기
# rag  활용!
def add_to_rag(col, all_traces):
    # 전체 과거 로그 -> 컬렉션에 추가
    docs, metas, ids = [], [], []
    for trace in all_traces:
        docs.append(preprocess(trace["msg"]))
        metas.append({"ts": trace["ts"], "host": trace["host"]})
        ids.append(str(trace["id"]))
    col.add(documents=docs, metadatas=metas, ids=ids)


def retrieve_context(col, candidates, k=3):
    # 후보 로그를 하나의 쿼리 문자열로 합쳐서 검색
    query = " ".join(preprocess(t["msg"]) for t in candidates)
    result = col.query(query_texts=[query], n_results=k)
    return result["documents"][0]  # 리스트 of strings


# llm 요약 및 판단
def llm_summarize(traces, context=None):
    text = "\n".join(f'{t["host"]}: {t["msg"]}' for t in traces)
    if context:
        prompt = (
            f"과거 유사 로그:\n{context}\n\n새 로그:\n{text}\n→ 한 문장으로 요약해줘."
        )
    else:
        prompt = f"다음 로그를 한 문장으로 요약해줘:\n{text}"
    return gemma.invoke(prompt).strip()


def llm_decide(summary):
    system = (
        "너는 SOC 자동화 에이전트다. 다음 형식 JSON만 반환:\n"
        '{ "action": "notify|ignore", "severity":"low|medium|high", "reason":"15자 이내" }'
    )
    resp = gpt.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"요약: {summary}"},
        ]
    ).content
    try:
        return json.loads(resp)
    except:
        return {"action": "ignore", "severity": "low", "reason": "json오류"}


# slack 알림
def notify_slack(decision, summary):
    msg = f"🚨 *{decision['severity'].upper()}* – {decision['reason']}\n```{summary}```"
    res = requests.post(SLACK_WEBHOOK, json={"text": msg})
    return res.status_code == 200


# 파이프라인
def run_pipeline():
    init_dummy_db()

    # 트레이스 로드 후 중복 제거
    traces = dedup(fetch_traces(limit=10))
    if not traces:
        print("로그 없음")
        return

    # ml 필터링 후 이상행위 후보 선정정
    candidates = filter_anomalies(traces, contamination=0.3)
    if not candidates:
        print("이상 후보 없음")
        return

    # rag 스토어 초기화 & 과거 로그 적재
    global RAG_COL
    try:
        RAG_COL  # 이미 초기화된 경우
    except NameError:
        RAG_COL = init_rag_store()
        add_to_rag(RAG_COL, dedup(fetch_traces(limit=1000)))  # 과거 1,000건 등록

    # 후보 -> 컨텍스트 검색
    context = retrieve_context(RAG_COL, candidates, k=3)

    # 요약 후 판단
    summary = llm_summarize(candidates, context=context)
    decision = llm_decide(summary)
    print("요약:", summary)
    print("판단:", decision)

    # 알림
    if decision.get("action") == "notify":
        ok = notify_slack(decision, summary)
        print("Slack 전송:", ok)
    else:
        print("알림 불필요")


if __name__ == "__main__":
    start = datetime.now()
    run_pipeline()
    print("소요시간:", datetime.now() - start)
