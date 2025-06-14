# sequential_agent test
"""
3개의 트레이스를 더미 db에서 읽어오기
먼저 중복 제거 -> gemma phi3로 여러줄 로그를 한문장 요약 -> gpt-4o로 notify / ignore 의사결정(json) -> notify일 경우 메세지 전송
"""

import os, json, sqlite3, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # .env → os.environment

DB_PATH = "traces.db"


def init_dummy_db():
    # 더미 db 설정, 나중에 실제 db 설계?
    if os.path.exists(DB_PATH):  # 만약 존재하면 return
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()  # sqlite3.row를 위한 전용 커서
    cursor.execute(
        "CREATE TABLE traces (id INTEGER PRIMARY KEY, ts TEXT, host TEXT, msg TEXT)"
    )
    demo = [
        ("2025-06-12 10:01:00", "srv1", "sshd: Failed password for root from 1.2.3.4"),
        ("2025-06-12 10:01:01", "srv1", "sshd: Failed password for root from 1.2.3.4"),
        ("2025-06-12 10:05:03", "srv2", "nginx: GET /etc/passwd HTTP/1.1"),
        ("2025-06-12 10:07:10", "srv3", "user bob executed sudo su"),
    ]
    cursor.executemany("INSERT INTO traces(ts,host,msg) VALUES (?,?,?)", demo)
    conn.commit()
    conn.close()


def fetch_traces(limit: int = 3):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def dedup(traces):
    seen = set()  # set 자료형 -> 중복 허용 x
    out = []  # 출력값
    for t in traces:
        sig = hash((t["host"], t["msg"]))
        if sig not in seen:
            seen.add(sig)
            out.append(t)
    return out


from langchain_community.llms import Ollama  # ollama 모델 사용

gemma = Ollama(model="gemma:2b-instruct-q4_0", temperature=0)


def llm_summarize(traces):
    text = "\n".join(f'{t["host"]}: {t["msg"]}' for t in traces)
    prompt = f"다음 로그들을 한국어 한 문장으로 요약해\n{text}"  # 데이터 요약
    return gemma.invoke(prompt).strip()  # llm이 반환한 문자열 반환


# gpt 4o 의사결정 -> json 형식으로
from langchain_openai import ChatOpenAI

gpt = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
)


def llm_decide(summary):
    system = (
        "너는 SOC 자동화 에이전트다. "
        "다음 형식의 JSON 데이터만 반환하라:\n"
        '{ "action": "notify|ignore", "severity":"low|medium|high", "reason": "15자 이내" }'
    )
    resp = gpt.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"요약: {summary}"},
        ]
    ).content
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return {"action": "ignore", "severity": "low", "reason": "json오류"}


# +++++ 추가 : slack 알림기능? <- 이건 안넣어도 됨
WEBHOOK = os.getenv("SLACK_WEBHOOK")


def notify_slack(decision, summary):
    msg = f"🚨 *{decision['severity'].upper()}* – {decision['reason']}\n```{summary}```"
    r = requests.post(WEBHOOK, json={"text": msg})
    return r.status_code == 200


def run_pipeline():  # 파이프라인 실행
    init_dummy_db()  # 더미 db 초기화
    traces = fetch_traces()  # 3개 트레이스를 가져와서
    traces = dedup(traces)  # 중복 제거후
    summary = llm_summarize(traces)  # 요약
    decision = llm_decide(summary)  # gpt 모델을 사용해서 의사결정
    print("요약:", summary)
    print("판단:", decision)

    # if decision["action"] == "notify":
    #     ok = notify_slack(decision, summary)
    #     print("Slack 전송:", ok)

    decisions = (
        decision if isinstance(decision, list) else [decision]
    )  # 의사결정이 리스트 형태가 아닐 경우 리스트로 변환
    # 여러개의 의사결정이 응답으로 올 경우 -> 각 요소를 리스트로 처리
    for decision in decisions:
        if decision.get("action") == "notify":
            ok = notify_slack(decision, summary)
            print("Slack 전송:", ok)
        else:
            print("알림 필요 없음")


if __name__ == "__main__":  # 프로그램이 실행되면 main함수 실행
    init_time = datetime.now()
    run_pipeline()
    print("소요시간\n", datetime.now() - init_time)
