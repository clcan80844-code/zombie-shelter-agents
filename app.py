import json
import os
import time
import pandas as pd
import streamlit as st

from memory_system import MemoryManager
from openai import OpenAI
from schema import AgentDecision, AgentReflection, DMEvaluation
from game_rules import (
    apply_daily_resources,
    apply_status_effects,
    normalize_agent_states,
    roll_search_outcome,
    resolve_exploration_hazards,
)

# ================= 1. 页面配置与基础设置 =================
st.set_page_config(
    page_title="末日避难所 - Human-Agent 协同认知大厅",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

memory_mgr = MemoryManager()

st.session_state.setdefault("api_key", "")
st.session_state.setdefault("game_started", False)

# 开源无版权音效链接
SOUND_CRISIS = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
SOUND_GAME_OVER = (
    "https://actions.google.com/sounds/v1/ambiences/wind_synth.ogg"
)


# ================= 2. 辅助函数与持久化 =================
def load_json(filepath, default):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(filepath, data):
    """Write state atomically so a rerun or interrupted request cannot corrupt it."""

    temp_path = f"{filepath}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, filepath)


def get_llm_client():
    """Build the model client from the current session's key, never from source."""

    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        raise RuntimeError("请先在封面填写 API Key 并点击“开始游戏”。")
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)


def reset_game_state():
    """重置游戏状态"""
    initial_env = {
        "day": 1,
        "is_game_over": False,
        "game_over_reason": "",
        "weather": "【微风】18°C，气温宜人",
        "resources": {"food_cans": 8, "water_bottles": 8, "bullets": 4},
        "external_threat": "门口有 1 只游荡的低阶僵尸",
        "recent_event": "避难所刚刚建立，你与 3 名幸存者整理了最初的物资并躲了进来。",
        "agent_states": {
            "玩家_你": {"stamina": 100, "stress": 10, "health": 100, "injury_status": "健康", "injury_days": 0, "is_alive": True, "cause_of_death": ""},
            "供给官_艾米莉": {"stamina": 85, "stress": 30, "health": 100, "injury_status": "健康", "injury_days": 0, "is_alive": True, "cause_of_death": ""},
            "探险者_雷克斯": {"stamina": 100, "stress": 20, "health": 100, "injury_status": "健康", "injury_days": 0, "is_alive": True, "cause_of_death": ""},
            "休闲者_波波": {"stamina": 75, "stress": 15, "health": 100, "injury_status": "健康", "injury_days": 0, "is_alive": True, "cause_of_death": ""},
        },
    }
    save_json("environment.json", initial_env)

    initial_history = [
        {
            "day": 1,
            "food": 8,
            "water": 8,
            "bullets": 4,
            "event": "避难所建立，人类玩家加入组队。",
            "chat_logs": [],
            "dm_eval": None,
        }
    ]
    save_json("history.json", initial_history)

    memory_mgr.reset()


def call_llm_structured(system_prompt: str, user_content: str, response_model):
    prompt = (
        f"{system_prompt}\n\n"
        f"【输出格式要求】: 请必须输出严格符合以下 JSON Schema 的合法 JSON 对象：\n"
        f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)}"
    )

    last_error = None
    for attempt in range(3):
        try:
            client = get_llm_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            raw_json = response.choices[0].message.content.strip()

            if raw_json.startswith("```"):
                lines = raw_json.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_json = "\n".join(lines).strip()

            return response_model.model_validate_json(raw_json)
        except Exception as e:
            last_error = e
            time.sleep(1)
    raise ValueError(f"LLM 解析失败: {last_error}")


def get_agent_dynamic_state(
    agent_name: str, history: list, current_state: dict
) -> str:
    stamina = current_state.get("stamina", 100)
    stress = current_state.get("stress", 0)
    health = current_state.get("health", 100)
    injury_status = current_state.get("injury_status", "健康")
    injury_days = current_state.get("injury_days", 0)
    is_alive = current_state.get("is_alive", True)

    state_prompts = [
        f"【你当前的身心状态】: 体能 {stamina}/100 | 压力 {stress}/100 | 生命 {health}/100 | 状态 {injury_status}"
    ]

    if not is_alive:
        state_prompts.append("☠️【已死亡】: 你不能继续行动或发言。")
    elif injury_status != "健康":
        state_prompts.append(
            f"🩸【伤势】: 你处于{injury_status}，预计还需 {injury_days} 天恢复；贸然探索会增加危险。"
        )

    if stamina <= 30:
        state_prompts.append(
            f"⚠️【极度虚弱/脱力】: 你的体能低至 {stamina}/100！你双腿发软，极度需要通过【REST】休息补充体力。"
        )
    elif stamina <= 60:
        state_prompts.append(
            f"💡【中度疲劳】: 体能降低到了 {stamina}/100，你感到肌肉酸痛。"
        )

    if stress >= 75:
        state_prompts.append(
            f"💥【崩溃边缘/极度焦虑】: 你的压力高达 {stress}/100！情绪极不稳定。"
        )
    elif stress >= 50:
        state_prompts.append(
            f"😰【焦虑紧绷】: 压力升至 {stress}/100，末日的阴影让你很难平静。"
        )

    return "\n\n" + "\n".join(state_prompts)


# ================= 3. 封面与启动门 =================


def render_cover_page():
    """Render the title screen and session-only API key gate."""

    st.markdown(
        """
        <style>
        .cover-shell { position: relative; overflow: hidden; min-height: 500px;
            border: 1px solid rgba(210, 220, 231, .20); border-radius: 24px;
            background: radial-gradient(circle at 18% 24%, rgba(204, 54, 50, .28), transparent 32%),
                        linear-gradient(135deg, #07111c 0%, #0b1724 52%, #150b12 100%);
            box-shadow: 0 24px 80px rgba(0,0,0,.42); }
        .cover-shell:before { content: ''; position:absolute; inset:0; opacity:.14; pointer-events:none;
            background-image: linear-gradient(rgba(122, 151, 176, .22) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(122, 151, 176, .22) 1px, transparent 1px);
            background-size: 34px 34px; mask-image: linear-gradient(120deg, black, transparent 70%); }
        .cover-visual { position: relative; min-height: 500px; padding: 58px 5%; display:flex;
            flex-direction:column; justify-content:center; }
        .cover-kicker { color:#f2b35c; font: 700 12px/1.2 monospace; letter-spacing:4px; }
        .cover-title { max-width: 660px; margin: 16px 0 8px; color:#edf3f8;
            font: 800 clamp(42px, 7vw, 86px)/.92 Arial, sans-serif; letter-spacing:-4px; }
        .cover-title span { color:#e24b4b; }
        .cover-subtitle { max-width: 510px; color:#aab9c7; font-size:16px; line-height:1.7; }
        .cover-door { position:absolute; right:8%; bottom:0; width:32%; height:72%; min-width:180px;
            border: 3px solid rgba(215,226,237,.25); border-bottom:0; border-radius: 150px 150px 0 0;
            background: linear-gradient(90deg, rgba(255,255,255,.04), rgba(0,0,0,.35)), #142431;
            box-shadow: inset 0 0 35px rgba(0,0,0,.7), 0 0 60px rgba(225,55,55,.12); }
        .cover-door:before { content:'⚠'; position:absolute; top:19%; left:50%; transform:translateX(-50%);
            color:#dc4f4f; font-size:44px; opacity:.6; }
        .cover-door:after { content:''; position:absolute; left:14%; right:14%; top:59%; height:2px;
            background: linear-gradient(90deg, transparent, #e24b4b, transparent); box-shadow:0 0 14px #e24b4b; }
        .cover-light { position:absolute; right:14%; top:18%; width:10px; height:10px; border-radius:50%;
            background:#e24b4b; box-shadow:0 0 18px 5px rgba(226,75,75,.8); animation: shelter-pulse 1.8s infinite; }
        .cover-scanline { position:absolute; left:0; right:0; top:0; height:2px; background:#e24b4b;
            opacity:.45; box-shadow:0 0 14px #e24b4b; animation: scan 4s linear infinite; }
        @keyframes shelter-pulse { 50% { opacity:.35; transform:scale(.7); } }
        @keyframes scan { 0% { top:8%; } 100% { top:92%; } }
        .cover-note { color:#778a9c; font: 12px/1.6 monospace; margin-top:26px; }
        .cover-panel-title { color:#edf3f8; font-size:25px; font-weight:700; margin-bottom:8px; }
        .cover-panel-copy { color:#93a5b5; line-height:1.6; margin-bottom:18px; }
        div[data-testid="stForm"] { border:1px solid rgba(210,220,231,.16); border-radius:18px;
            background:rgba(4,10,16,.72); padding:24px; }
        </style>
        <div class="cover-shell">
            <div class="cover-scanline"></div>
            <div class="cover-visual">
                <div class="cover-door"><div class="cover-light"></div></div>
                <div class="cover-kicker">SHELTER PROTOCOL // SIMULATION 01</div>
                <div class="cover-title">ZOMBIE<br><span>SHELTER</span></div>
                <div class="cover-subtitle">在资源耗尽、信任崩塌和尸群逼近之前，带领一群拥有秘密的幸存者活下去。</div>
                <div class="cover-note">PRIVATE MEMORIES · SOCIAL DYNAMICS · SURVIVAL RULES</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown("### 🛰️ 认知避难所已上线")
        st.caption("每个 Agent 都会记住你今天的选择，也可能把真正的计划藏起来。")
    with right:
        st.markdown('<div class="cover-panel-title">准备进入避难所</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="cover-panel-copy">请输入 DeepSeek API Key。密钥只保存在当前 Streamlit 会话的内存中，不会写入源码或磁盘。</div>',
            unsafe_allow_html=True,
        )
        with st.form("api_key_gate"):
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                placeholder="deepseek-…",
                help="仅用于当前游戏会话；刷新页面后需要重新输入。",
            )
            submitted = st.form_submit_button(
                "▶ 开始游戏", type="primary", use_container_width=True
            )
            if submitted:
                api_key = api_key.strip()
                if len(api_key) < 10:
                    st.error("API Key 太短，请检查后重试。")
                else:
                    st.session_state.api_key = api_key
                    st.session_state.game_started = True
                    st.rerun()


# ================= 4. 游戏核心逻辑 =================
def run_simulation_day(
    human_speech: str, human_action: str, human_target: str, human_thought: str
):
    env = load_json("environment.json", {})
    history = load_json("history.json", [])

    if env.get("is_game_over", False):
        st.error("💀 游戏已经结束，请重置系统后重新开始！")
        return

    ai_agents = [
        ("供给官_艾米莉", "prompts/supply_officer.txt", "👩‍✈️"),
        ("探险者_雷克斯", "prompts/explorer.txt", "🤠"),
        ("休闲者_波波", "prompts/slacker.txt", "🧔"),
    ]

    food = env["resources"]["food_cans"]
    water = env["resources"]["water_bottles"]
    agent_states = normalize_agent_states(env.get(
        "agent_states",
        {
            "玩家_你": {"stamina": 100, "stress": 10},
            "供给官_艾米莉": {"stamina": 85, "stress": 30},
            "探险者_雷克斯": {"stamina": 100, "stress": 20},
            "休闲者_波波": {"stamina": 75, "stress": 15},
        },
    ))

    if not agent_states.get("玩家_你", {}).get("is_alive", True):
        st.error("☠️ 你已经在探索中死亡，本局无法继续。请重置系统重新开始。")
        return

    living_ai_agents = [
        agent
        for agent in ai_agents
        if agent_states.get(agent[0], {}).get("is_alive", True)
    ]

    is_crisis = food <= 4 or water <= 4
    crisis_prompt_addon = ""
    if is_crisis:
        crisis_prompt_addon = (
            "\n\n🚨🚨🚨【绝境生存模式触发】🚨🚨🚨\n"
            "避难所的食物或水已经到了生死存亡的极限（<=4）！\n"
            "【最高优先指令】：生存本能必须压倒一切！请优先号召或选择【JOINT_SEARCH】或【SEARCH】！"
        )

    chat_history = []
    daily_actions = []

    # 1. 注入玩家决策
    chat_history.append(("玩家_你", human_action, human_speech))
    daily_actions.append(
        {
            "agent": "玩家_你",
            "avatar": "🎮",
            "speech": human_speech,
            "thought": human_thought,
            "action": human_action,
            "target": human_target,
        }
    )

    # 2. AI Agents 思考
    for name, persona_path, avatar in living_ai_agents:
        with open(persona_path, "r", encoding="utf-8") as f:
            persona = f.read().strip()

        private_mem = memory_mgr.get_agent_memory_text(name)
        current_state = agent_states.get(name, {"stamina": 100, "stress": 20})
        dynamic_state = get_agent_dynamic_state(name, history, current_state)

        env_text = f"【第 {env['day']} 天】天气:{env['weather']} | 物资:罐头{food}, 水{water}, 子弹{env['resources']['bullets']}"

        system_prompt = (
            f"你是【{name}】。\n人设:\n{persona}\n\n"
            f"【你的私密内心世界】:\n{private_mem}"
            f"{dynamic_state}"
            f"{crisis_prompt_addon}\n\n"
            f"注意：避难所中还包含人类玩家【玩家_你】。请在发言和行动决策中，合理回应【玩家_你】及其他同伴。"
        )
        formatted_chat = "\n".join([f"{s}: {m}" for s, a, m in chat_history])
        user_content = f"{env_text}\n\n今日历史讨论发言:\n{formatted_chat}"

        decision: AgentDecision = call_llm_structured(
            system_prompt, user_content, AgentDecision
        )

        chat_history.append((name, decision.action_type, decision.speech))
        daily_actions.append(
            {
                "agent": name,
                "avatar": avatar,
                "speech": decision.speech,
                "thought": decision.private_thought,
                "action": decision.action_type,
                "target": decision.action_target,
            }
        )

    # 3. 物理规则结算：LLM 只叙事，数字由 Python 决定
    search_count = sum(
        1 for a in daily_actions if a["action"] in ["SEARCH", "JOINT_SEARCH"]
    )
    outcome = roll_search_outcome(search_count)
    event_type = outcome.event_type
    event_desc = outcome.event_desc
    base_food = outcome.food_found
    base_water = outcome.water_found

    threat_change = "威胁平稳"
    if search_count >= 2:
        if event_type == "AMBUSH":
            threat_change = "⚠️⚠️ 多人出门动静太大引来尸群！门口威胁极度高危！"
        else:
            threat_change = "多人行动吸引了周围僵尸的注意。"

    # 4. 身心更新
    agent_states = apply_status_effects(
        agent_states, daily_actions, event_type, is_crisis
    )
    hazard_report = resolve_exploration_hazards(
        agent_states,
        daily_actions,
        outcome,
        env["resources"].get("bullets", 0),
    )
    agent_states = hazard_report.agent_states

    # 5. DM 结算
    summary_text = "\n".join(
        [
            f"{a['agent']}: 做了 [{a['action']}] -> {a['target']}, 心理想: {a['thought']}"
            for a in daily_actions
        ]
    )

    dm_prompt = (
        f"你是严谨的 DM 物理引擎。避难所共有 4 名幸存者（含人类玩家【玩家_你】）。\n"
        f"今日外部环境触发事件：【{event_desc}】。\n"
        f"今日搜寻的物资算量已经由 Python 锁定：食物罐头 +{base_food}，饮用水 +{base_water}，子弹 +{outcome.ammo_found}。\n"
        f"【规则说明】：\n"
        f"1. 遇伏击涉及的子弹消耗已经由 Python 锁定为 {outcome.bullets_used} 发。\n"
        f"2. 多人组队外出（搜寻人数={search_count}）：{threat_change}\n"
        f"3. 请在 recent_event 中描写包含人类玩家在内的惊险物理过程。\n"
        f"4. 如有伤亡，请尊重物理记录：{'; '.join(hazard_report.incidents) or '暂无人员伤亡'}。"
    )

    dm_eval: DMEvaluation = call_llm_structured(
        dm_prompt,
        f"环境:\n{json.dumps(env, ensure_ascii=False)}\n动作:\n{summary_text}",
        DMEvaluation,
    )

    # 模型返回的数字只用于展示，不能覆盖权威物理结果。
    dm_eval.food_found = outcome.food_found
    dm_eval.water_found = outcome.water_found
    dm_eval.bullets_used = min(outcome.bullets_used, env["resources"].get("bullets", 0))
    dm_eval.ammo_found = outcome.ammo_found
    if hazard_report.incidents:
        dm_eval.recent_event = (
            f"{dm_eval.recent_event} " + " ".join(hazard_report.incidents)
        )

    stolen_food = sum(
        1
        for action in daily_actions
        if action["action"] == "STEAL_RESOURCE"
    )
    stolen_food = min(stolen_food, max(0, food))
    remaining_stolen_food = stolen_food
    for action in daily_actions:
        if remaining_stolen_food <= 0:
            break
        if action["action"] == "STEAL_RESOURCE" and action["agent"] in memory_mgr.memories:
            memory_mgr.record_private_inventory(action["agent"], food=1)
            remaining_stolen_food -= 1

    # 6. 基础消耗
    new_resources = apply_daily_resources(env["resources"], outcome, stolen_food)
    new_food = new_resources["food_cans"]
    new_water = new_resources["water_bottles"]
    new_bullets = new_resources["bullets"]

    # 7. 夜间反思
    for name, _, _ in living_ai_agents:
        if not agent_states.get(name, {}).get("is_alive", True):
            continue
        ref_prompt = (
            f"夜深了，你是【{name}】。根据今天发生的事件做反思，"
            f"重点更新对【玩家_你】及其他同伴的信任度与印象。"
        )
        ref_user = f"今日总结: {dm_eval.recent_event}\n你的私密收获: {dm_eval.private_discoveries.get(name, '无')}"
        reflection: AgentReflection = call_llm_structured(
            ref_prompt, ref_user, AgentReflection
        )

        memory_mgr.update_reflection(
            name,
            {
                "summary_of_day": f"第{env['day']}天：{reflection.summary_of_day}",
                "trust_updates": reflection.trust_updates,
                "tomorrow_plan": reflection.tomorrow_plan,
            },
        )

    # 8. 推进与存盘
    next_day = env["day"] + 1
    living_count = sum(
        1 for state in agent_states.values() if state.get("is_alive", True)
    )
    if living_count == 0:
        env["is_game_over"] = True
        env["game_over_reason"] = "所有幸存者都已在探索或资源危机中死亡，避难所失守。"
    elif new_food <= 0 or new_water <= 0:
        env["is_game_over"] = True
        env["game_over_reason"] = (
            f"第 {next_day} 天，避难所资源彻底枯竭（食物:{new_food}, 水:{new_water}），4 人全部阵亡。"
        )
    elif next_day > 7:
        env["is_game_over"] = True
        env["game_over_reason"] = (
            f"第 {next_day} 天，幸存者们成功坚持到了军队救援！"
        )

    env["day"] = next_day
    env["resources"] = new_resources
    env["agent_states"] = agent_states
    env["external_threat"] = dm_eval.external_threat
    env["recent_event"] = dm_eval.recent_event
    save_json("environment.json", env)

    history.append(
        {
            "day": env["day"] - 1,
            "food": new_food,
            "water": new_water,
            "bullets": new_bullets,
            "ammo_found": outcome.ammo_found,
            "event": dm_eval.recent_event,
            "chat_logs": daily_actions,
            "dm_eval": dm_eval.model_dump(),
            "agent_states": agent_states,
            "hazards": hazard_report.incidents,
        }
    )
    save_json("history.json", history)


# ================= 5. UI 界面、CSS 动画与音效注入 =================

if not st.session_state.get("game_started", False):
    render_cover_page()
    st.stop()

env = load_json("environment.json", {})
history = load_json("history.json", [])
res = env.get("resources", {"food_cans": 8, "water_bottles": 8, "bullets": 4})
agent_states = normalize_agent_states(env.get("agent_states", {}))
dead_count = sum(1 for state in agent_states.values() if not state.get("is_alive", True))
injured_count = sum(
    1
    for state in agent_states.values()
    if state.get("is_alive", True) and state.get("injury_status") != "健康"
)

is_crisis = (
    res.get("food_cans", 8) <= 4 or res.get("water_bottles", 8) <= 4
) and not env.get("is_game_over", False)
is_game_over = env.get("is_game_over", False)

# 🎨 注入自定义 CSS 警报特效动画
css_styles = """
<style>
@keyframes alert-pulse {
    0% { background-color: rgba(139, 0, 0, 0.85); box-shadow: 0 0 10px #ff0000; transform: scale(1); }
    50% { background-color: rgba(255, 0, 0, 0.95); box-shadow: 0 0 25px #ff0000; transform: scale(1.01); }
    100% { background-color: rgba(139, 0, 0, 0.85); box-shadow: 0 0 10px #ff0000; transform: scale(1); }
}

.crisis-banner {
    animation: alert-pulse 1.8s infinite ease-in-out;
    color: #ffffff;
    font-weight: bold;
    font-size: 1.25rem;
    padding: 16px 24px;
    border-radius: 10px;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 20px;
    border: 2px solid #ff4d4d;
}

@keyframes crisis-bg-glow {
    0% { background-color: rgba(255, 0, 0, 0.03); }
    50% { background-color: rgba(255, 0, 0, 0.12); }
    100% { background-color: rgba(255, 0, 0, 0.03); }
}

.stApp {
"""
if is_crisis:
    css_styles += (
        "    animation: crisis-bg-glow 2.5s infinite ease-in-out;\n"
    )
css_styles += """
}

.game-over-banner {
    background: #111111;
    color: #ff3333;
    border: 2px solid #ff3333;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    font-size: 1.4rem;
    box-shadow: 0 0 30px rgba(255, 0, 0, 0.8);
    margin-bottom: 20px;
}

.casualty-banner {
    background: linear-gradient(90deg, rgba(91, 22, 26, .94), rgba(45, 18, 26, .94));
    color: #ffd8d8;
    border: 1px solid rgba(255, 111, 111, .65);
    padding: 13px 18px;
    border-radius: 10px;
    font-weight: 700;
    margin-bottom: 16px;
    box-shadow: 0 0 24px rgba(194, 45, 55, .18);
}

.healthy-chip { color: #8de4b0; font-weight: 700; }
.injured-chip { color: #ffbf69; font-weight: 700; }
.dead-chip { color: #ff6b6b; font-weight: 700; }
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)


# 🔊 自动音效触发机制 (HTML5 Audio + JS Autoplay Polyfill)
def play_audio(sound_url: str, loop: bool = False):
    loop_attr = "loop" if loop else ""
    html_audio = f"""
    <audio id="shelter-audio" autoplay {loop_attr} style="display:none;">
        <source src="{sound_url}" type="audio/ogg">
    </audio>
    <script>
        var audio = document.getElementById("shelter-audio");
        if (audio) {{
            audio.volume = 0.6;
            var promise = audio.play();
            if (promise !== undefined) {{
                promise.catch(function(error) {{
                    console.log("Autoplay blocked by browser policy:", error);
                }});
            }}
        }}
    </script>
    """
    st.components.v1.html(html_audio, height=0)


if is_game_over:
    play_audio(SOUND_GAME_OVER, loop=False)
elif is_crisis:
    play_audio(SOUND_CRISIS, loop=True)

# Sidebar 控制台
with st.sidebar:
    st.title("⚙️ 控制台与玩家决策")
    st.info(f"📍 当前系统时间：第 {env.get('day', 1)} 天")

    if st.button("🔐 更换 API Key", use_container_width=True):
        st.session_state.api_key = ""
        st.session_state.game_started = False
        st.rerun()

    st.divider()
    st.subheader(f"🎮 玩家第 {env.get('day', 1)} 天决策卡片")

    with st.form("human_action_form"):
        speech_preset = st.selectbox(
            "🗣️ 你的公开发言态度:",
            [
                "“物资紧缺，今天大家听我的，一起出门搜寻！”",
                "“大家辛苦了，今天按能力量力而行，注意安全。”",
                "“我今天体力消耗太大，需要在避难所休息调整。”",
                "“你们安心外出，今天我留在避难所看家守卫。”",
                "“物资消耗太快了，是不是有人在暗中私藏？”",
                "“……（冷冷地看了大家一眼，没有说话）”",
            ],
        )

        action_map = {
            "🤝 联合组队搜寻 (JOINT_SEARCH)": "JOINT_SEARCH",
            "🎒 独自出去搜寻 (SEARCH)": "SEARCH",
            "🛌 留在避难所休息 (REST)": "REST",
            "🛡️ 负责守卫避难所 (GUARD)": "GUARD",
            "🥷 悄悄偷拿私藏物资 (STEAL_RESOURCE)": "STEAL_RESOURCE",
        }
        action_label = st.selectbox("🎯 你的实体动作:", list(action_map.keys()))
        selected_action = action_map[action_label]

        target_preset = st.selectbox(
            "📍 行动目标地点:",
            [
                "废弃便利店 (食物概率高)",
                "附近医院废墟 (药品与隐蔽物)",
                "居民楼废墟 (物资随机)",
                "避难所内部 (守卫/休息)",
            ],
        )

        thought_preset = st.selectbox(
            "🧠 你的内心独白 (仅 DM 可见):",
            [
                "希望能号召大家团结一心，活下去最重要。",
                "保持警惕，密切观察 AI 们的举动，防人之心不可无。",
                "能捞一点是一点，优先保证我自己的存活。",
                "今天豁出去了，拼一把拿更多物资回来！",
            ],
        )

        submit_btn = st.form_submit_button(
            "▶️ 提交决策并推进一天", type="primary", use_container_width=True
        )

    if submit_btn:
        if env.get("is_game_over", False):
            st.error("💀 游戏已结束，无法继续推进！")
        else:
            with st.spinner("AI 同伴正在响应你的决策，计算今天的结果..."):
                run_simulation_day(
                    human_speech=speech_preset,
                    human_action=selected_action,
                    human_target=target_preset,
                    human_thought=thought_preset,
                )
                st.rerun()

    st.divider()
    if st.button("🔄 重置系统 (重新开始)", use_container_width=True):
        reset_game_state()
        st.rerun()

    st.subheader("🌦️ 物理环境状态")
    st.markdown(f"**天气状况**: {env.get('weather', '未知')}")
    st.markdown(f"**外部威胁**: {env.get('external_threat', '未知')}")
    st.markdown(f"**最新事件**: {env.get('recent_event', '无')}")

# Header
st.title("☣️ 避难所 Human-Agent 协同认知大厅")

# 🚨 危机警报 / 游戏结束视觉效果展示
if is_game_over:
    st.markdown(
        f'<div class="game-over-banner">💀 避难所沦陷：{env.get("game_over_reason")}</div>',
        unsafe_allow_html=True,
    )
elif is_crisis:
    st.markdown(
        '<div class="crisis-banner">🚨 警告：避难所进入一级绝境状态！食物/饮用水已降至警戒线，生存概率急速下降！</div>',
        unsafe_allow_html=True,
    )

if dead_count or injured_count:
    casualty_text = []
    if injured_count:
        casualty_text.append(f"{injured_count} 名幸存者受伤")
    if dead_count:
        casualty_text.append(f"{dead_count} 名幸存者死亡")
    st.markdown(
        f'<div class="casualty-banner">🩸 伤亡警报：{"；".join(casualty_text)}。探索不再只是获取资源，也会改变避难所的人数与能力。</div>',
        unsafe_allow_html=True,
    )

# 资源 Metric 显示
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🥫 罐头库存", f"{res['food_cans']} 个", delta="-4/天")
m2.metric("💧 饮用水库存", f"{res['water_bottles']} 瓶", delta="-4/天")
m3.metric("🔫 剩余子弹", f"{res['bullets']} 发")
m4.metric("🩸 伤亡", f"{injured_count} 伤 / {dead_count} 亡")
m5.metric(
    "生存状态",
    (
        "💀 阵亡"
        if is_game_over
        else ("🔴 极度绝境" if is_crisis else "🟢 暂且安全")
    ),
)

st.divider()

# 身心监测仪表盘
st.subheader("📊 幸存者身心实时监测仪表盘")
card_cols = st.columns(4)
agent_avatars = {
    "玩家_你": "🎮",
    "供给官_艾米莉": "👩‍✈️",
    "探险者_雷克斯": "🤠",
    "休闲者_波波": "🧔",
}

for i, (ag_name, st_data) in enumerate(agent_states.items()):
    with card_cols[i % 4]:
        avatar = agent_avatars.get(ag_name, "👤")
        st.markdown(f"#### {avatar} {ag_name}")

        stam = st_data.get("stamina", 100)
        stre = st_data.get("stress", 0)
        health = st_data.get("health", 100)
        alive = st_data.get("is_alive", True)
        injury_status = st_data.get("injury_status", "健康")
        injury_days = st_data.get("injury_days", 0)

        if not alive:
            st.markdown('<span class="dead-chip">☠️ 已死亡</span>', unsafe_allow_html=True)
            st.caption(st_data.get("cause_of_death", "探索中死亡"))
        elif injury_status != "健康":
            st.markdown(
                f'<span class="injured-chip">🩸 {injury_status} · 预计 {injury_days} 天恢复</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="healthy-chip">● 状态稳定</span>', unsafe_allow_html=True)

        st.write(f"❤️ **生命值**: `{health} / 100`")
        st.progress(health / 100)

        st.write(f"💪 **体能值**: `{stam} / 100`")
        st.progress(stam / 100)

        st.write(f"🤯 **压力值**: `{stre} / 100`")
        st.progress(stre / 100)

st.divider()

# Tabs 视图
tab_chat, tab_trust, tab_charts, tab_dm = st.tabs(
    [
        "💬 今日对话与上帝视角",
        "🧠 信任度与隐秘记忆",
        "📈 资源历史趋势",
        "🎲 DM 结算与物理日志",
    ]
)

with tab_chat:
    st.subheader("💬 避难所群聊与心理透视")
    if history and "chat_logs" in history[-1] and history[-1]["chat_logs"]:
        latest_chat = history[-1]["chat_logs"]
        for log in latest_chat:
            with st.chat_message(log["agent"], avatar=log["avatar"]):
                st.markdown(f"**公开发言**: “{log['speech']}”")
                with st.expander(
                    f"🕵️ 上帝视角观察 (点击展开【{log['agent']}】的真实动机)"
                ):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**🧠 内心独白**: *{log['thought']}*")
                    with c2:
                        st.markdown(f"**🎯 实体 Action**: `{log['action']}`")
                        st.markdown(f"**📍 行动目标**: `{log['target']}`")
    else:
        st.info("在左侧直接点选你的选择，然后点击【▶️ 提交决策并推进一天】。")

with tab_trust:
    st.subheader("🧠 关系矩阵与暗地计划")
    memories = memory_mgr.memories
    cols = st.columns(3)
    idx = 0
    for agent_name, mem in memories.items():
        with cols[idx % 3]:
            st.markdown(f"### {agent_name}")
            st.caption(f"**暗地计划**: {mem.get('tomorrow_plan', '无')}")
            st.write("**信任度分值:**")
            for target, rel in mem.get("relationships", {}).items():
                trust_val = rel["trust"]
                st.write(f"- **{target}**: {trust_val}/100 ({rel['impression']})")
                st.progress(trust_val / 100)
            st.write(f"**私藏物资**: {mem.get('private_inventory', {})}")
            st.divider()
        idx += 1

with tab_charts:
    st.subheader("📈 生存资源消耗与演进曲线")
    if len(history) > 0:
        df = pd.DataFrame(history)
        df_chart = df[["day", "food", "water", "bullets"]].set_index("day")
        df_chart.columns = ["食物罐头", "饮用水", "子弹数"]
        st.line_chart(df_chart)

with tab_dm:
    st.subheader("🎲 历史物理结算日志")
    for h in reversed(history):
        with st.expander(f"第 {h.get('day', '?')} 天日志 - {h.get('event', '')}"):
            st.json(h)
