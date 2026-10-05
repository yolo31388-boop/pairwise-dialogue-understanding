"""
游戏内对话理解与自然语言处理系统 - 核心模块

提供完整的对话理解流水线：
- understand_input(): 输入理解（分词 + 词性标注）
- intent_recognition(): 意图识别（分类 + 置信度）
- entity_extraction(): 实体提取（类型 + 关系）
- dialogue_state(): 对话状态跟踪（上下文 + 槽位）
- response_generation(): 回复生成（模板 + 个性化）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


# ---------------------------------------------------------------------------
# 词典
# ---------------------------------------------------------------------------

_PLACES = {"铁匠铺", "酒馆", "药店", "王城", "森林", "新手村", "矿洞", "市场", "城堡"}
_PERSONS = {"铁匠", "村长", "商人", "守卫", "法师", "艾尔文"}
_ITEMS = {"铁剑", "长剑", "药水", "盾牌", "铠甲", "面包", "宝石", "火把", "匕首"}
_VERBS = {"买", "卖", "购买", "出售", "去", "走", "在", "位于", "有", "要", "想",
          "接", "找", "完成", "接受", "攻击"}
_PRONOUNS = {"我", "你", "您", "他", "她", "我们", "大家"}
_QUESTION_WORDS = {"哪里", "哪儿", "什么", "怎么", "怎样", "谁", "多少", "多少钱", "几"}
_PARTICLES = {"吗", "呢", "吧", "啊", "的", "了", "嘛"}
_GREETINGS = {"你好", "您好", "嗨", "早上好", "晚上好", "喂"}
_FAREWELLS = {"再见", "拜拜", "回头见"}
_NUMERALS = set("一二两三四五六七八九十百千万")

_POS_LEXICON: Dict[str, str] = {}
for _w in _PLACES | _PERSONS | _ITEMS:
    _POS_LEXICON[_w] = "名词"
for _w in _VERBS:
    _POS_LEXICON[_w] = "动词"
for _w in _PRONOUNS:
    _POS_LEXICON[_w] = "代词"
for _w in _QUESTION_WORDS:
    _POS_LEXICON[_w] = "疑问词"
for _w in _PARTICLES:
    _POS_LEXICON[_w] = "语气词"
for _w in _GREETINGS | _FAREWELLS:
    _POS_LEXICON[_w] = "叹词"

_LEXICON = set(_POS_LEXICON) | _NUMERALS
_MAX_WORD_LEN = 4

_ENTITY_TYPES = {}
for _w in _PLACES:
    _ENTITY_TYPES[_w] = "place"
for _w in _PERSONS:
    _ENTITY_TYPES[_w] = "person"
for _w in _ITEMS:
    _ENTITY_TYPES[_w] = "item"

# ---------------------------------------------------------------------------
# 意图模型
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: Dict[str, Dict[str, float]] = {
    "greeting": {"你好": 2.0, "您好": 2.0, "嗨": 1.5, "早上好": 2.0, "晚上好": 2.0, "喂": 1.0},
    "farewell": {"再见": 2.0, "拜拜": 2.0, "回头见": 2.0},
    "ask_location": {"哪里": 2.5, "哪儿": 2.5, "怎么": 1.0, "怎么去": 2.5, "怎么走": 2.5},
    "buy": {"买": 2.0, "购买": 2.0, "多少钱": 1.5},
    "sell": {"卖": 2.0, "出售": 2.0},
    "quest": {"任务": 2.0, "委托": 2.0, "接受": 1.0, "接": 1.0},
    "quest_complete": {"完成": 2.0, "交任务": 2.0},
    "threat": {"杀": 2.0, "攻击": 2.0, "打": 1.0},
}

_INTENT_SLOTS: Dict[str, List[str]] = {
    "buy": ["item", "quantity"],
    "sell": ["item", "quantity"],
    "ask_location": ["destination"],
    "quest": ["quest"],
}

# ---------------------------------------------------------------------------
# 回复模板（按个性化风格区分）
# ---------------------------------------------------------------------------

_RESPONSE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "greeting": {
        "friendly": "你好呀，{player}！今天想做点什么？",
        "gruff": "嗯，你好。有事快说。",
        "formal": "您好，{player}。请问有何贵干？",
    },
    "farewell": {
        "friendly": "再见啦，{player}，欢迎再来！",
        "gruff": "走吧走吧。",
        "formal": "告辞了，{player}，祝您旅途顺利。",
    },
    "ask_location": {
        "friendly": "{destination}就在{hint}，很好找的！",
        "gruff": "{destination}？{hint}，自己去。",
        "formal": "{destination}位于{hint}，{player}。",
    },
    "buy": {
        "friendly": "好嘞，{quantity}个{item}，给您包好了！",
        "gruff": "{item}，{quantity}个。拿钱来。",
        "formal": "这是您要的{item}，共{quantity}件，请收好。",
    },
    "sell": {
        "friendly": "我看看……{item}是吧，给您算个好价钱！",
        "gruff": "{item}？行，我收了。",
        "formal": "您要出售的{item}，我们愿意收购。",
    },
    "quest": {
        "friendly": "太感谢了！这个任务就拜托你了！",
        "gruff": "任务给你，别搞砸了。",
        "formal": "感谢您的援手，{player}，委托详情在此。",
    },
    "quest_complete": {
        "friendly": "哇，你真的完成了！这是给你的奖励！",
        "gruff": "完成了？哼，还算有点本事。",
        "formal": "任务完成得非常出色，{player}，这是您的报酬。",
    },
    "threat": {
        "friendly": "别、别动手！有话好说！",
        "gruff": "想打架？我奉陪！",
        "formal": "请您冷静，暴力解决不了问题。",
    },
    "unknown": {
        "friendly": "唔……我没太听懂，能再说一遍吗？",
        "gruff": "说人话。",
        "formal": "抱歉，我未能理解您的意思。",
    },
}

_ASK_SLOT_TEMPLATES = {
    "item": "你想交易什么物品？",
    "quantity": "要多少数量呢？",
    "destination": "你想去哪里？",
    "quest": "你想了解哪个任务？",
}

_PLACE_HINTS = {
    "铁匠铺": "村子东边", "酒馆": "广场旁边", "药店": "市场街角",
    "王城": "北方大道尽头", "森林": "村子西边", "新手村": "传送门附近",
    "矿洞": "山脚下", "市场": "城镇中心", "城堡": "王城中央",
}

_DEFAULT_PERSONALITY = "friendly"


class DialogueUnderstanding:
    """游戏内对话理解与自然语言处理系统。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._npcs: Dict[str, str] = self.config.get("npcs", {})
        self._state = {}
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 输入理解：分词 + 词性标注
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """基于词典的正向最大匹配分词。"""
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isspace() or ch in "，。！？；：、,.!?;:\"'“”‘’（）()":
                i += 1
                continue
            if ch.isascii() and (ch.isalnum()):
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalnum():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            matched = None
            for length in range(min(_MAX_WORD_LEN, len(text) - i), 1, -1):
                piece = text[i:i + length]
                if piece in _LEXICON:
                    matched = piece
                    break
            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(ch)
                i += 1
        return tokens

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """为每个词标注词性。"""
        tagged = []
        for token in tokens:
            if token in _POS_LEXICON:
                tagged.append((token, _POS_LEXICON[token]))
            elif token in _NUMERALS or token.isdigit():
                tagged.append((token, "数词"))
            else:
                tagged.append((token, "名词"))
        return tagged

    def understand_input(self, text: str, speaker: str = "player") -> Dict:
        """完整理解一句输入：分词、词性、意图、实体，并更新对话状态。"""
        tokens = self.tokenize(text)
        pos = self.pos_tag(tokens)
        intent = self.intent_recognition(text)
        extraction = self.entity_extraction(text)
        result = {
            "text": text,
            "tokens": tokens,
            "pos": pos,
            "intent": intent,
            "entities": extraction["entities"],
            "relations": extraction["relations"],
        }
        self._advance_state(result, speaker)
        return result

    # ------------------------------------------------------------------
    # 意图识别：分类 + 置信度
    # ------------------------------------------------------------------

    def intent_recognition(self, text: str) -> Dict:
        """基于加权评分对输入进行意图分类，并给出置信度。"""
        scores = {intent: 0.0 for intent in _INTENT_KEYWORDS}
        for intent, keywords in _INTENT_KEYWORDS.items():
            for keyword, weight in keywords.items():
                if keyword in text:
                    scores[intent] += weight
        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]
        if best_score <= 0.0:
            return {"intent": "unknown", "confidence": 0.0, "scores": scores}
        confidence = best_score / (best_score + 1.0)
        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    # ------------------------------------------------------------------
    # 实体提取：类型 + 关系
    # ------------------------------------------------------------------

    def entity_extraction(self, text: str) -> Dict:
        """提取文本中的实体（人名/地名/物品名）及实体间关系。"""
        entities = []
        occupied = [False] * len(text)
        for word, entity_type in sorted(_ENTITY_TYPES.items(),
                                        key=lambda kv: -len(kv[0])):
            start = text.find(word)
            while start != -1:
                end = start + len(word)
                if not any(occupied[start:end]):
                    entities.append({
                        "text": word,
                        "type": entity_type,
                        "start": start,
                    })
                    for i in range(start, end):
                        occupied[i] = True
                start = text.find(word, start + 1)
        entities.sort(key=lambda e: e["start"])

        persons = [e["text"] for e in entities if e["type"] == "person"]
        places = [e["text"] for e in entities if e["type"] == "place"]
        items = [e["text"] for e in entities if e["type"] == "item"]

        relations = []
        subject = persons[0] if persons else ("玩家" if "我" in text else None)
        if ("在" in text or "位于" in text) and persons and places:
            relations.append({"subject": persons[0], "predicate": "位于",
                              "object": places[0]})
        if ("去" in text or "走" in text) and places:
            relations.append({"subject": subject or "玩家", "predicate": "前往",
                              "object": places[0]})
        if ("买" in text or "购买" in text) and items:
            relations.append({"subject": subject or "玩家", "predicate": "购买",
                              "object": items[0]})
        if ("卖" in text or "出售" in text) and items:
            relations.append({"subject": subject or "玩家", "predicate": "出售",
                              "object": items[0]})
        return {"entities": entities, "relations": relations}

    # ------------------------------------------------------------------
    # 对话状态：上下文 + 槽位
    # ------------------------------------------------------------------

    def _advance_state(self, understanding: Dict, speaker: str):
        intent = understanding["intent"]["intent"]
        entities = understanding["entities"]
        raw_intent = intent
        if intent == "unknown" and self._state.get("current_intent") in _INTENT_SLOTS:
            # 未识别出新意图时，沿用上一轮意图继续填槽（多轮对话）
            intent = self._state["current_intent"]

        slots = self._state.get("slots", {})
        if intent in _INTENT_SLOTS:
            intent_slots = slots.setdefault(intent,
                                            {s: None for s in _INTENT_SLOTS[intent]})
            for entity in entities:
                if entity["type"] == "item" and "item" in intent_slots:
                    intent_slots["item"] = entity["text"]
                if entity["type"] == "place" and "destination" in intent_slots:
                    intent_slots["destination"] = entity["text"]
            for token, tag in understanding["pos"]:
                if tag == "数词" and "quantity" in intent_slots:
                    intent_slots["quantity"] = token
        self._state["slots"] = slots
        self._state["current_intent"] = intent

        self._history.append({
            "speaker": speaker,
            "text": understanding["text"],
            "intent": raw_intent,
            "entities": [e["text"] for e in entities],
            "time": time.time(),
        })

    def dialogue_state(self) -> Dict:
        """返回当前对话状态：轮次、当前意图、槽位填充情况与上下文。"""
        intent = self._state.get("current_intent")
        slots = self._state.get("slots", {}).get(intent, {}) if intent else {}
        missing = [name for name, value in slots.items() if value is None]
        return {
            "turn": len(self._history),
            "current_intent": intent,
            "slots": dict(slots),
            "missing_slots": missing,
            "context": list(self._history),
        }

    # ------------------------------------------------------------------
    # 回复生成：模板 + 个性化
    # ------------------------------------------------------------------

    def _personality_of(self, npc_id: Optional[str]) -> str:
        personality = self._npcs.get(npc_id, _DEFAULT_PERSONALITY)
        if personality not in ("friendly", "gruff", "formal"):
            return _DEFAULT_PERSONALITY
        return personality

    def response_generation(self, npc_id: Optional[str] = None,
                            player_name: str = "冒险者") -> str:
        """根据当前对话状态，用模板生成带个性化风格的回复。"""
        state = self.dialogue_state()
        intent = state["current_intent"] or "unknown"
        personality = self._personality_of(npc_id)
        templates = _RESPONSE_TEMPLATES.get(intent, _RESPONSE_TEMPLATES["unknown"])
        template = templates[personality]

        slots = dict(state["slots"])
        if intent in _INTENT_SLOTS:
            for missing in state["missing_slots"]:
                if missing in _ASK_SLOT_TEMPLATES:
                    return _ASK_SLOT_TEMPLATES[missing]
        values = {
            "player": player_name,
            "item": slots.get("item") or "东西",
            "quantity": slots.get("quantity") or "一",
            "destination": slots.get("destination") or "那里",
            "hint": _PLACE_HINTS.get(slots.get("destination"), "不远的地方"),
        }
        try:
            return template.format(**values)
        except KeyError:
            return template
