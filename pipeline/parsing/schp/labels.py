# Look Into Person (LIP) dataset labels and mapping

LIP_CLASSES = [
    'background',      # 0
    'hat',             # 1
    'hair',            # 2
    'glove',           # 3
    'sunglasses',      # 4
    'upper-clothes',   # 5
    'dress',           # 6
    'coat',            # 7
    'socks',           # 8
    'pants',           # 9
    'torso-skin',      # 10
    'scarf',           # 11
    'skirt',           # 12
    'face',            # 13
    'left-arm',        # 14
    'right-arm',       # 15
    'left-leg',        # 16
    'right-leg',       # 17
    'left-shoe',       # 18
    'right-shoe'       # 19
]

# ID to Label string mapping
LIP_LABELS = {i: name for i, name in enumerate(LIP_CLASSES)}

# Label string to ID mapping
LIP_LABEL_TO_ID = {name: i for i, name in enumerate(LIP_CLASSES)}
