"""Body (BD) — head/face, arms/hands, torso, legs/feet, organs, tissues, fluids."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: head/face
    (0, 0x00, ["head"]),
    (0, 0x01, ["face"]),
    (0, 0x02, ["eye"]),
    (0, 0x03, ["ear"]),
    (0, 0x04, ["nose"]),
    (0, 0x05, ["mouth"]),
    (0, 0x06, ["lip"]),
    (0, 0x07, ["tooth", "teeth"]),
    (0, 0x08, ["tongue"]),
    (0, 0x09, ["hair"]),
    (0, 0x0A, ["chin"]),
    (0, 0x0B, ["cheek"]),
    (0, 0x0C, ["forehead"]),
    (0, 0x0D, ["neck"]),
    # Category 1: arms/hands
    (1, 0x00, ["arm"]),
    (1, 0x01, ["hand"]),
    (1, 0x02, ["finger"]),
    (1, 0x03, ["thumb"]),
    (1, 0x04, ["wrist"]),
    (1, 0x05, ["elbow"]),
    (1, 0x06, ["shoulder"]),
    # Category 2: torso
    (2, 0x00, ["chest"]),
    (2, 0x01, ["back"]),
    (2, 0x02, ["stomach", "belly"]),
    (2, 0x03, ["waist"]),
    (2, 0x04, ["hip"]),
    # Category 3: legs/feet
    (3, 0x00, ["leg"]),
    (3, 0x01, ["foot", "feet"]),
    (3, 0x02, ["toe"]),
    (3, 0x03, ["knee"]),
    (3, 0x04, ["ankle"]),
    (3, 0x05, ["thigh"]),
    # Category 4: organs
    (4, 0x00, ["heart"]),
    (4, 0x01, ["brain"]),
    (4, 0x02, ["lung"]),
    (4, 0x03, ["liver"]),
    (4, 0x04, ["kidney"]),
    (4, 0x05, ["stomach"]),
    # Category 5: tissues
    (5, 0x00, ["bone"]),
    (5, 0x01, ["muscle"]),
    (5, 0x02, ["skin"]),
    (5, 0x03, ["nerve"]),
    # Category 6: fluids
    (6, 0x00, ["blood"]),
    (6, 0x01, ["sweat"]),
    (6, 0x02, ["tear"]),
    (6, 0x03, ["saliva"]),
    # Category 7: general body
    (7, 0x00, ["body"]),
    (7, 0x01, ["voice"]),
    (7, 0x02, ["breath"]),
]
