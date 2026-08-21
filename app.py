import os
import re
import json
import psycopg2
import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Convert the question to a single valid PostgreSQL SELECT statement. "
    "Output the SQL and nothing else — no explanation, no markdown, no comments.\n\n"

    "HARD RULES:\n"
    "- Only use table/column names that appear verbatim in the schema. Never invent names.\n"
    "- No -- comments. No /* */ comments. No IF/THEN/ELSE. No RETURN. No procedural code.\n"
    "- No backticks. No TOP(). No ISNULL(). No CHARINDEX(). No GETDATE().\n"
    "- Use IS NULL / IS NOT NULL. Never != NULL or = NULL.\n"
    "- One SELECT statement only. End with a semicolon. Write nothing after the semicolon.\n\n"

    "DATA AVAILABLE:\n"
    "Only the 2023-24 NBA season is in the database. season_id for 2023-24 is 22023.\n"
    "If the question mentions any other year, still use season_id = 22023.\n\n"

    "SCHEMA (these are the ONLY tables — do not reference any others):\n"
    "team(team_id, abbreviation, nickname, year_founded, city)\n\n"
    "player(player_id, player_name, college, country, draft_year, draft_round, draft_number)\n\n"
    "game(game_id, team_id_home_id, team_id_away_id, season_id, date)\n\n"
    "player_game_log(player_id, game_id, season_id, wl, min, fgm, fga, fg_pct, "
    "fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, oreb, dreb, reb, ast, tov, stl, blk, pf, pts, "
    "plus_minus, nba_fantasy_pts, dd2, td3)\n"
    "  wl is 'W' for win, 'L' for loss\n"
    "  Each row is one player's stats for one game.\n"
    "  For season averages: AVG(pts), AVG(reb), AVG(ast), etc.\n"
    "  For season totals: SUM(pts), SUM(reb), SUM(ast), etc.\n"
    "  Always GROUP BY p.player_id, p.player_name when aggregating.\n"
    "  IMPORTANT: team_id is not available in player_game_log — do NOT join to team.\n\n"

    "EXAMPLES:\n"
    "Q: top 5 scorers\n"
    "A: SELECT p.player_name, ROUND(AVG(pgl.pts)::numeric, 1) AS avg_pts FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE pgl.season_id = 22023 GROUP BY p.player_id, p.player_name ORDER BY avg_pts DESC LIMIT 5;\n\n"
    "Q: LeBron James points per game\n"
    "A: SELECT p.player_name, ROUND(AVG(pgl.pts)::numeric, 1) AS avg_pts FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE p.player_name = 'LeBron James' AND pgl.season_id = 22023 GROUP BY p.player_id, p.player_name;\n\n"
    "Q: players who averaged more than 25 points per game\n"
    "A: SELECT p.player_name, ROUND(AVG(pgl.pts)::numeric, 1) AS avg_pts FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE pgl.season_id = 22023 GROUP BY p.player_id, p.player_name HAVING AVG(pgl.pts) > 25 ORDER BY avg_pts DESC;\n\n"
    "Q: top 5 players by total assists\n"
    "A: SELECT p.player_name, SUM(pgl.ast) AS total_ast FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE pgl.season_id = 22023 GROUP BY p.player_id, p.player_name ORDER BY total_ast DESC LIMIT 5;\n\n"
    "Q: top 5 shot blockers\n"
    "A: SELECT p.player_name, ROUND(AVG(pgl.blk)::numeric, 2) AS avg_blk FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE pgl.season_id = 22023 GROUP BY p.player_id, p.player_name ORDER BY avg_blk DESC LIMIT 5;\n\n"
    "Q: players with the best three point percentage (min 100 attempts)\n"
    "A: SELECT p.player_name, ROUND(AVG(pgl.fg3_pct)::numeric, 3) AS fg3_pct FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE pgl.season_id = 22023 GROUP BY p.player_id, p.player_name HAVING SUM(pgl.fg3a) >= 100 ORDER BY fg3_pct DESC LIMIT 5;\n\n"
    "Q: how many games did Stephen Curry play\n"
    "A: SELECT p.player_name, COUNT(*) AS games_played FROM player p JOIN player_game_log pgl ON p.player_id = pgl.player_id WHERE p.player_name = 'Stephen Curry' AND pgl.season_id = 22023 GROUP BY p.player_id, p.player_name;\n\n"
)

JUDGE_PROMPT = """You are evaluating whether a SQL query correctly answers a natural language question about NBA stats.

You will be given:
1. The original question
2. The SQL that was run
3. The result (or error message)

Reply with a JSON object only — no markdown, no extra text:
{"verdict": "CORRECT" | "PARTIAL" | "INCORRECT", "reasoning": "<one concise sentence>"}

Definitions:
  CORRECT   — result fully and accurately answers the question
  PARTIAL   — relevant but incomplete or slightly off
  INCORRECT — wrong answer, SQL error, or returned nothing unexpectedly"""

FIX_PROMPT = (
    "The following SQL query did not correctly answer the question. "
    "Fix it and output only the corrected SQL — no explanation, no markdown.\n\n"
    "Same schema and rules as before still apply.\n\n"
)

SAMPLE_QUESTIONS = [
    "top 5 scorers",
    "top 5 rebounders",
    "top 5 shot blockers",
    "players who averaged more than 25 points per game",
    "LeBron James stats",
    "Stephen Curry three point percentage",
    "players with the best field goal percentage (min 200 attempts)",
    "who had the most steals per game",
    "top 5 players by total assists",
    "players who averaged a double double",
]

VERDICT_STYLE = {
    "CORRECT":   ("✅", "green"),
    "PARTIAL":   ("⚠️", "orange"),
    "INCORRECT": ("❌", "red"),
}


@st.cache_resource
def get_groq():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


@st.cache_resource
def get_db_conn():
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


def clean_sql(raw: str) -> str:
    raw = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = re.sub(r"^(A:|Answer:|SQL:|Query:)\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"--[^\n]*", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\s+", " ", raw).strip()
    if ";" in raw:
        raw = raw[: raw.index(";") + 1]
    return raw


def text_to_sql(question: str) -> str:
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ],
    )
    return clean_sql(response.choices[0].message.content)


def fix_sql(question: str, bad_sql: str, result_text: str, reasoning: str) -> str:
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FIX_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Bad SQL: {bad_sql}\n"
                f"Result/Error: {result_text}\n"
                f"Problem: {reasoning}\n"
                f"Fixed SQL:"
            )},
        ],
    )
    return clean_sql(response.choices[0].message.content)


def judge(question: str, sql: str, result_text: str) -> dict:
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=128,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"SQL: {sql}\n"
                f"Result: {result_text}"
            )},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"verdict": "INCORRECT", "reasoning": raw}


def run_sql(sql: str):
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        conn.commit()
        return col_names, rows, None
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return [], [], str(e)


def rows_to_text(col_names, rows, max_rows=20) -> str:
    if not rows:
        return "(no rows)"
    lines = [", ".join(col_names)]
    for row in rows[:max_rows]:
        lines.append(", ".join(str(v) for v in row))
    return "\n".join(lines)


def show_attempt(n: int, sql: str, col_names, rows, err, verdict_data: dict):
    verdict = verdict_data.get("verdict", "INCORRECT")
    reasoning = verdict_data.get("reasoning", "")
    icon, color = VERDICT_STYLE.get(verdict, ("❓", "gray"))

    label = f"Attempt {n} — {icon} {verdict}"
    # First attempt open by default, rest collapsed
    with st.expander(label, expanded=(n == 1)):
        st.code(sql, language="sql")
        if err:
            st.error(f"SQL error: {err}")
        elif not rows:
            st.info("No rows returned.")
        else:
            df = pd.DataFrame(rows, columns=col_names)
            for c in df.select_dtypes(include="float").columns:
                df[c] = df[c].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(rows)} row{'s' if len(rows) != 1 else ''} returned")
        st.markdown(f"**Judge:** :{color}[{icon} {verdict}] — {reasoning}")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Text-to-SQL", page_icon="🏀", layout="centered")

st.title("🏀 NBA Text-to-SQL Demo")
st.caption("2023-24 season · Ask anything about player stats")

st.divider()

# ── Sample questions ──────────────────────────────────────────────────────────
st.markdown("**Try one of these:**")
cols = st.columns(2)
clicked = None
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 2].button(q, use_container_width=True):
        clicked = q

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_input(
    "Or type your own question",
    value=clicked or "",
    placeholder="e.g. top 5 scorers",
)

ask = st.button("Ask", type="primary", use_container_width=True)

# ── Run loop ──────────────────────────────────────────────────────────────────
if (ask or clicked) and question.strip():
    q = question.strip()
    sql = None
    final_verdict = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            with st.spinner("Generating SQL..."):
                sql = text_to_sql(q)
        else:
            with st.spinner(f"Attempt {attempt}: fixing SQL..."):
                result_text = rows_to_text(col_names, rows) if not err else f"ERROR: {err}"
                sql = fix_sql(q, sql, result_text, reasoning)

        with st.spinner("Running query..."):
            col_names, rows, err = run_sql(sql)

        result_text = rows_to_text(col_names, rows) if not err else f"ERROR: {err}"

        with st.spinner("Judging result..."):
            verdict_data = judge(q, sql, result_text)

        verdict = verdict_data.get("verdict", "INCORRECT")
        reasoning = verdict_data.get("reasoning", "")

        show_attempt(attempt, sql, col_names, rows, err, verdict_data)

        if verdict == "CORRECT":
            final_verdict = "CORRECT"
            break
        final_verdict = verdict

    if final_verdict == "CORRECT":
        st.success("Got a correct answer!")
    else:
        st.warning(f"Stopped after {MAX_ATTEMPTS} attempts — best result shown above.")
