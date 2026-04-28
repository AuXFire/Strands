"""Emotion (EM) — basic, attachment, surprise, confidence, energy, social, complex.

Each entry: (category, concept, seed_words).
"""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: basic emotions
    (0, 0x00, ["happy", "joyful", "cheerful", "glad"]),
    (0, 0x01, ["sad", "unhappy", "sorrowful", "mournful"]),
    (0, 0x02, ["angry", "furious", "irate", "enraged"]),
    (0, 0x03, ["afraid", "scared", "frightened", "terrified"]),
    (0, 0x04, ["calm", "peaceful", "serene", "tranquil"]),
    (0, 0x05, ["disgusted", "repulsed", "revolted"]),
    (0, 0x06, ["jealous", "envious"]),
    (0, 0x07, ["guilty", "ashamed", "remorseful"]),
    # Category 1: attachment
    (1, 0x00, ["love", "adore", "cherish"]),
    (1, 0x01, ["hate", "detest", "despise", "loathe"]),
    (1, 0x02, ["like", "fond", "favor"]),
    (1, 0x03, ["dislike", "averse"]),
    (1, 0x04, ["affection", "warmth", "tenderness"]),
    (1, 0x05, ["devotion", "loyalty", "fidelity"]),
    # Category 2: surprise
    (2, 0x00, ["surprise", "astonish", "amaze", "astound"]),
    (2, 0x01, ["shock", "stun", "stagger"]),
    (2, 0x02, ["wonder", "marvel", "awe"]),
    # Category 3: confidence
    (3, 0x00, ["confident", "assured", "certain"]),
    (3, 0x01, ["doubt", "uncertain", "unsure"]),
    (3, 0x02, ["proud", "prideful"]),
    (3, 0x03, ["humble", "modest"]),
    (3, 0x04, ["embarrassed", "humiliated"]),
    # Category 4: energy
    (4, 0x00, ["excited", "thrilled", "enthusiastic", "elated"]),
    (4, 0x01, ["bored", "tedious", "dull"]),
    (4, 0x02, ["tired", "exhausted", "weary", "fatigued"]),
    (4, 0x03, ["energetic", "vigorous", "lively"]),
    (4, 0x04, ["relaxed", "rested"]),
    # Category 5: social emotions
    (5, 0x00, ["lonely", "isolated", "solitary"]),
    (5, 0x01, ["friendly", "amiable", "cordial"]),
    (5, 0x02, ["hostile", "aggressive", "antagonistic"]),
    (5, 0x03, ["compassion", "sympathy", "empathy"]),
    (5, 0x04, ["kindness", "kind", "benevolent"]),
    (5, 0x05, ["cruel", "mean", "malicious"]),
    # Category 6: complex
    (6, 0x00, ["nostalgia", "nostalgic"]),
    (6, 0x01, ["melancholy", "wistful"]),
    (6, 0x02, ["anxious", "anxiety", "worried", "nervous"]),
    (6, 0x03, ["hope", "hopeful", "optimistic"]),
    (6, 0x04, ["despair", "hopeless", "despondent"]),
    (6, 0x05, ["content", "satisfied", "fulfilled"]),
    (6, 0x06, ["frustrated", "annoyed", "irritated"]),
]
