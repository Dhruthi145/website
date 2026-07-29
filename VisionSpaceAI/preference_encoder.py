"""
Module: preference_encoder.py
Step 8: Preference Encoding — expanded with wall color, curtains, decor, lighting, flooring, plants
"""
import numpy as np

STYLES = ['modern', 'minimalist', 'bohemian', 'scandinavian',
          'industrial', 'traditional', 'mid_century', 'luxury', 'rustic']

ROOM_TYPES = ['living_room', 'bedroom', 'kitchen', 'bathroom',
              'dining_room', 'office', 'kids_room', 'hallway']

COLOR_THEMES = ['neutral', 'warm', 'cool', 'monochrome',
                'earthy', 'pastel', 'bold', 'dark']

WALL_COLORS = [
    'white', 'off_white', 'warm_beige', 'greige', 'light_grey', 'charcoal',
    'navy_blue', 'sage_green', 'forest_green', 'dusty_rose', 'terracotta',
    'mustard_yellow', 'deep_teal', 'blush_pink', 'slate_blue', 'warm_taupe',
    'cream', 'black', 'olive_green', 'burgundy'
]

CURTAIN_STYLES = ['none', 'sheer_white', 'linen_natural', 'velvet_rich',
                  'blackout_dark', 'patterned_boho', 'silk_elegant', 'cotton_casual']

WALL_DECOR = ['none', 'minimal_frames', 'gallery_wall', 'large_mirror',
              'floating_shelves', 'tapestry', 'abstract_art', 'botanical_prints',
              'geometric_panels', 'neon_sign']

LIGHTING_MOODS = ['natural_daylight', 'warm_ambient', 'cool_bright',
                  'dramatic_spot', 'candlelight_warm', 'neon_accent', 'golden_hour']

FLOORING = ['keep_existing', 'light_oak_wood', 'dark_walnut_wood', 'white_marble',
            'grey_concrete', 'plush_carpet', 'herringbone_wood', 'terracotta_tiles',
            'black_white_checkered', 'chevron_parquet']

PLANTS = ['none', 'minimal_one_plant', 'few_plants', 'lush_jungle']

CEILING_STYLES = ['standard_white', 'exposed_wooden_beams', 'coffered_classic',
                  'modern_cove_lighting', 'dark_painted', 'industrial_exposed']

BUDGET_MIN = 500.0
BUDGET_MAX = 50000.0


def one_hot(value: str, categories: list) -> np.ndarray:
    vec = np.zeros(len(categories), dtype=np.float32)
    if value in categories:
        vec[categories.index(value)] = 1.0
    return vec


def normalize_budget(budget: float) -> float:
    return float(np.clip((budget - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN), 0.0, 1.0))


def encode_preferences(preferences: dict) -> np.ndarray:
    vecs = [
        one_hot(preferences.get('style',         'modern'),          STYLES),
        one_hot(preferences.get('room_type',      'living_room'),     ROOM_TYPES),
        one_hot(preferences.get('color_theme',    'neutral'),         COLOR_THEMES),
        one_hot(preferences.get('wall_color',     'white'),           WALL_COLORS),
        one_hot(preferences.get('curtain_style',  'none'),            CURTAIN_STYLES),
        one_hot(preferences.get('wall_decor',     'none'),            WALL_DECOR),
        one_hot(preferences.get('lighting_mood',  'warm_ambient'),    LIGHTING_MOODS),
        one_hot(preferences.get('flooring',       'keep_existing'),   FLOORING),
        one_hot(preferences.get('add_plants',     'none'),            PLANTS),
        one_hot(preferences.get('ceiling_style',  'standard_white'),  CEILING_STYLES),
        np.array([normalize_budget(float(preferences.get('budget', 5000)))], dtype=np.float32),
    ]
    return np.concatenate(vecs)
