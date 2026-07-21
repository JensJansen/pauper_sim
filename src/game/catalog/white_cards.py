"""White-identity card catalog. Empty for now -- no implemented deck plays
a white card yet. Same shape as every other color file once populated:
a WHITE_CARD_CATALOG dict (name -> CardDef) and a WHITE_EFFECT_REGISTRY
dict (EffectId -> spec), unioned into game.CARD_DEFS/EFFECT_REGISTRY by
game/registry.py."""

WHITE_CARD_CATALOG = {}
WHITE_EFFECT_REGISTRY = {}
