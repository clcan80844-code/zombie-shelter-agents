import json
import os


class MemoryManager:

    def __init__(self, filepath="agent_memories.json"):
        self.filepath = filepath
        self.memories = self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        default_data = self._default_data()
        self._save(default_data)
        return default_data

    @staticmethod
    def _default_data():
        """Return a fresh memory graph for a new run."""

        return {
            "供给官_艾米莉": {
                "relationships": {
                    "探险者_雷克斯": {
                        "trust": 60,
                        "impression": "冲动但有战斗力",
                    },
                    "休闲者_波波": {"trust": 30, "impression": "偷懒且消耗物资"},
                },
                "private_inventory": {"hidden_food": 0, "hidden_water": 0},
                "episodic_memory": ["第1天：避难所建立，需要控制物资。"],
                "tomorrow_plan": "监督大家节约水资源。",
            },
            "探险者_雷克斯": {
                "relationships": {
                    "供给官_艾米莉": {"trust": 70, "impression": "掌管物资，比较苛刻"},
                    "休闲者_波波": {"trust": 40, "impression": "拖后腿的人"},
                },
                "private_inventory": {"hidden_food": 0, "hidden_water": 0},
                "episodic_memory": ["第1天：我是主要的战斗和搜寻战力。"],
                "tomorrow_plan": "争取出门搜寻，证明价值。",
            },
            "休闲者_波波": {
                "relationships": {
                    "供给官_艾米莉": {
                        "trust": 40,
                        "impression": "总是不给我足够的吃的",
                    },
                    "探险者_雷克斯": {
                        "trust": 50,
                        "impression": "看起来很凶但好忽悠",
                    },
                },
                "private_inventory": {"hidden_food": 0, "hidden_water": 0},
                "episodic_memory": ["第1天：活下去就行，尽量别干重活。"],
                "tomorrow_plan": "混一天是一天，找机会攒点私房食物。",
            },
        }
    def reset(self):
        """Reset both the on-disk file and this manager's in-memory cache."""

        self.memories = self._default_data()
        self._save()

    def _save(self, data=None):
        if data is None:
            data = self.memories
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_agent_memory_text(self, agent_name: str) -> str:
        agent_data = self.memories.get(agent_name, {})
        rel_str = "\n".join(
            [
                f"  - {target}: 信任度 {info['trust']}/100, 印象: '{info['impression']}'"
                for target, info in agent_data.get("relationships", {}).items()
            ]
        )
        mem_str = "\n".join(
            [f"  - {log}" for log in agent_data.get("episodic_memory", [])[-3:]]
        )  # 最近3条记忆

        return (
            f"【对其他人的关系矩阵与信任度】:\n{rel_str}\n"
            f"【私藏物资】: {agent_data.get('private_inventory', {})}\n"
            f"【近期记忆碎片】:\n{mem_str}\n"
            f"【暗地里的计划】: {agent_data.get('tomorrow_plan', '无')}"
        )

    def update_reflection(self, agent_name: str, reflection_data: dict):
        if agent_name not in self.memories:
            return
        agent = self.memories[agent_name]

        # 1. 追加短期事件记忆
        agent["episodic_memory"].append(reflection_data["summary_of_day"])

        # 2. 更新信任度关系
        for target, delta in reflection_data.get("trust_updates", {}).items():
            if target in agent["relationships"]:
                current = agent["relationships"][target]["trust"]
                agent["relationships"][target]["trust"] = max(
                    0, min(100, current + delta)
                )

        # 3. 更新明日计划
        agent["tomorrow_plan"] = reflection_data.get("tomorrow_plan", "")
        self._save()

    def record_private_inventory(self, agent_name: str, food: int = 0, water: int = 0):
        """Record a bounded private stash created by a concrete game action."""

        if agent_name not in self.memories:
            return
        inventory = self.memories[agent_name].setdefault("private_inventory", {})
        inventory["hidden_food"] = max(0, int(inventory.get("hidden_food", 0)) + food)
        inventory["hidden_water"] = max(0, int(inventory.get("hidden_water", 0)) + water)
        self._save()
