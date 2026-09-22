# Zombie Shelter Agents

一个带有人机协作、私密记忆、关系信任和资源生存规则的多 Agent 避难所模拟游戏。

## 启动

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

启动后会先进入封面。API Key 通过密码框输入，只保存在当前 Streamlit 会话内存中，不会写入源码或 JSON 存档。

命令行版本 `main.py` 使用环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的 Key"
python main.py
```

## 当前架构

- `app.py`：Streamlit 入口、人机交互和游戏展示
- `game_rules.py`：确定性的物理规则、资源结算和身心状态更新
- `memory_system.py`：Agent 的关系矩阵、事件记忆和私藏物资
- `schema.py`：LLM 输出的 Pydantic 结构与边界校验
- `main.py`：命令行模拟器

模型负责提出行动、表达观点和生成叙事；Python 负责资源数字、状态边界和游戏结束判定。

## 生存机制

- 探索可能触发轻伤、重伤或死亡；生命值和恢复倒计时会持续影响下一回合。
- 伏击风险最高；弹药充足时会降低伤害，弹药耗尽时伤害会进一步上升。
- 搜索可能找到额外子弹：普通搜寻有中等概率获得少量子弹，大丰收事件有较高概率获得更多子弹。
- 伤亡会出现在顶部警报、幸存者仪表盘、历史 DM 日志和资源结算中。
