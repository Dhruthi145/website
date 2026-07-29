"""
Module: prompt_generator.py
Step 10: Prompt Generation

Generates rich, detailed prompts — 150-200 tokens is FINE now because
design_generator.py uses compel / SD 2.1 / SDXL, none of which have a
77-token limit. More detail = better output.
"""
import numpy as np

STYLE = {
    'modern':       'sleek contemporary interior, clean geometric lines, minimal ornament, polished surfaces',
    'minimalist':   'ultra-minimalist interior, deliberate empty space, monochrome tones, only essential furniture',
    'bohemian':     'boho-chic eclectic interior, layered textiles, macrame, mixed warm patterns, global accents',
    'scandinavian': 'Scandinavian hygge interior, pale birch wood, white walls, cozy textiles, functional simplicity',
    'industrial':   'urban industrial loft, exposed brick, raw steel beams, concrete, Edison bulb pendants',
    'traditional':  'classic traditional interior, ornate crown moulding, rich mahogany wood, antique accents',
    'mid_century':  'mid-century modern, organic curved forms, teak furniture, mustard and olive palette, retro geometry',
    'luxury':       'ultra-luxury interior, Calacatta marble, brushed brass fixtures, bespoke velvet furniture, opulent',
    'rustic':       'cosy rustic farmhouse, reclaimed barn wood, natural stone fireplace, linen, vintage patina',
}

WALL_COLOR = {
    'white':           'pure white painted walls, crisp and airy',
    'off_white':       'off-white warm ivory painted walls, soft and creamy',
    'warm_beige':      'warm beige painted walls, sandy honey tone',
    'greige':          'greige walls, grey-beige blend, sophisticated neutral',
    'light_grey':      'light silver-grey painted walls, cool and modern',
    'charcoal':        'deep charcoal grey painted walls, dramatic and bold',
    'navy_blue':       'rich navy blue painted walls, deep and luxurious',
    'sage_green':      'soft sage green painted walls, calming botanical tone',
    'forest_green':    'deep forest green painted walls, earthy and rich',
    'dusty_rose':      'dusty rose painted walls, romantic and soft',
    'terracotta':      'warm terracotta clay painted walls, Mediterranean warmth',
    'mustard_yellow':  'golden mustard yellow painted walls, bold and cheerful',
    'deep_teal':       'deep teal painted walls, jewel-toned sophistication',
    'blush_pink':      'soft blush pink painted walls, delicate and feminine',
    'slate_blue':      'muted slate blue painted walls, calm and coastal',
    'warm_taupe':      'warm taupe painted walls, earthy and grounding',
    'cream':           'creamy warm white painted walls, soft and luminous',
    'black':           'matte black painted walls, bold editorial statement',
    'olive_green':     'olive green painted walls, earthy organic warmth',
    'burgundy':        'deep burgundy wine-red painted walls, rich and moody',
}

CURTAIN = {
    'none':            None,
    'sheer_white':     'floor-to-ceiling sheer white voile curtains gently diffusing soft natural light',
    'linen_natural':   'natural undyed linen curtains with relaxed gathered drape, raw organic texture',
    'velvet_rich':     'floor-length rich velvet curtains, heavy sumptuous fabric, formal elegant drape',
    'blackout_dark':   'thick blackout curtains in charcoal grey, tailored pinch-pleat, complete light control',
    'patterned_boho':  'boho patterned curtains with printed ethnic geometric motif, warm earthy colours',
    'silk_elegant':    'silk dupioni curtains with soft sheen, pinch-pleated, luxurious floor-length drape',
    'cotton_casual':   'casual cotton tab-top curtains, relaxed unpretentious style, light airy fabric',
}

DECOR = {
    'none':             None,
    'minimal_frames':   'two to three minimal thin black-framed art prints with simple line art, evenly spaced on wall',
    'gallery_wall':     'curated gallery wall with mix of framed photos, art prints and small mirrors in matching gold frames',
    'large_mirror':     'one oversized statement mirror with thin gold frame mounted on wall, reflecting light',
    'floating_shelves': 'clean floating white wooden shelves mounted on wall, styled with candles, books and ceramic vases',
    'tapestry':         'large handwoven textile tapestry hung on wall, boho artisanal fibre art',
    'abstract_art':     'single large-scale abstract canvas painting with bold brush strokes, dominant wall art',
    'botanical_prints': 'set of three framed vintage botanical illustration prints, herbarium style, matching frames',
    'geometric_panels': 'decorative 3D geometric wall panels creating textured architectural surface detail',
    'neon_sign':        'custom neon light sign mounted on wall, soft warm glow, contemporary accent piece',
}

LIGHTING = {
    'warm_ambient':     'warm 2700K amber ambient lighting, glowing table and floor lamps, cosy intimate atmosphere',
    'natural_daylight': 'flooded with bright soft natural daylight through large windows, airy and open',
    'cool_bright':      'crisp bright 5000K cool white lighting, uniform even illumination, clean and energising',
    'dramatic_spot':    'dramatic directional spotlights creating chiaroscuro contrast, deep shadows and bright highlights',
    'candlelight_warm': '2200K candlelight-warm ultra-cozy glow, flickering warmth, intimate and romantic',
    'neon_accent':      'cool white base lighting with colourful LED neon accent strips, atmospheric modern glow',
    'golden_hour':      'golden hour sunlight pouring through windows at low angle, long warm shadows, magical warmth',
}

FLOORING = {
    'keep_existing':        None,
    'light_oak_wood':       'pale light oak hardwood flooring, Scandinavian blonde wide planks, visible natural grain',
    'dark_walnut_wood':     'rich dark walnut hardwood flooring, deep warm chocolate brown wide planks',
    'white_marble':         'polished white Carrara marble floor with elegant grey veining, reflective luxury surface',
    'grey_concrete':        'smooth polished concrete floor, matte mid-grey, industrial modern finish',
    'plush_carpet':         'deep pile plush neutral carpet, soft underfoot, warm and cosy',
    'herringbone_wood':     'light wood herringbone parquet flooring, classic Parisian elegance',
    'terracotta_tiles':     'handmade terracotta clay floor tiles, warm red-orange tone, rustic Mediterranean',
    'black_white_checkered':'classic black and white checkered ceramic floor tiles, retro graphic statement',
    'chevron_parquet':      'chevron pattern light wood parquet floor, geometric precision',
}

PLANTS = {
    'none':              None,
    'minimal_one_plant': 'one tall statement fiddle-leaf fig tree in a minimal white ceramic pot in the corner',
    'few_plants':        'a few carefully placed indoor plants including trailing pothos, snake plant and small succulents',
    'lush_jungle':       'lush indoor plant jungle with large monstera, tall palm, hanging ferns, trailing vines',
}

CEILING = {
    'standard_white':       None,
    'exposed_wooden_beams': 'ceiling with warm rustic exposed wooden beams, architectural character',
    'coffered_classic':     'coffered ceiling with decorative recessed panel grid, classical architectural elegance',
    'modern_cove_lighting': 'modern ceiling with integrated perimeter cove LED lighting, soft indirect ambient glow',
    'dark_painted':         'ceiling painted in a deep moody contrasting tone, dramatic overhead statement',
    'industrial_exposed':   'raw exposed concrete ceiling with visible ductwork, urban industrial loft aesthetic',
}

ROOM = {
    'living_room': 'spacious living room',
    'bedroom':     'serene bedroom',
    'kitchen':     'functional kitchen',
    'bathroom':    'spa-like bathroom',
    'dining_room': 'elegant dining room',
    'office':      'productive home office',
    'kids_room':   'playful kids bedroom',
    'hallway':     'welcoming entrance hallway',
}

THEME = {
    'neutral':     'balanced neutral color harmony, refined and timeless',
    'warm':        'warm color palette with gold and amber undertones, cozy and inviting',
    'cool':        'cool color palette with light blue and silver undertones, crisp and fresh',
    'earthy':      'organic earthy color palette with terracotta and moss accents, natural grounding',
    'monochrome':  'sophisticated monochrome color scheme with varying shades and textures',
    'pastel':      'soft pastel color palette with airy mint and blush accents, light and playful',
    'bold':        'bold high-contrast color palette with vibrant statement accents, energetic and modern',
    'dark':        'moody dark color palette with charcoal and deep wood tones, dramatic and cozy',
}

def _budget(norm):
    if norm < 0.2:  return 'budget-friendly IKEA-style pieces, cost-effective materials'
    if norm < 0.5:  return 'mid-range quality furniture, solid craftsmanship, durable materials'
    if norm < 0.8:  return 'high-quality furniture, designer-inspired pieces, premium materials throughout'
    return 'luxury bespoke furniture, top designer brands, finest premium finishes'


def generate_prompt(fused_features: np.ndarray,
                    preferences: dict,
                    detections: list) -> tuple:
    """
    Build full rich prompt — no token limit concerns.
    compel / SD 2.1 / SDXL handle 150-500+ tokens natively.
    """
    room    = ROOM.get(preferences.get('room_type', 'living_room'), 'living room')
    style   = STYLE.get(preferences.get('style', 'modern'), 'modern interior')
    theme   = THEME.get(preferences.get('color_theme', 'neutral'), 'natural color harmony')
    wcolor  = WALL_COLOR.get(preferences.get('wall_color', 'white'), 'white walls')
    curtain = CURTAIN.get(preferences.get('curtain_style', 'none'))
    decor   = DECOR.get(preferences.get('wall_decor', 'none'))
    light   = LIGHTING.get(preferences.get('lighting_mood', 'warm_ambient'), 'warm lighting')
    floor   = FLOORING.get(preferences.get('flooring', 'keep_existing'))
    plant   = PLANTS.get(preferences.get('add_plants', 'none'))
    ceil    = CEILING.get(preferences.get('ceiling_style', 'standard_white'))
    budget  = _budget((float(preferences.get('budget', 5000)) - 500) / 49500)
    objs    = ', '.join(d['label'] for d in detections[:6]) if detections else 'furniture'
    other   = (preferences.get('other_items') or preferences.get('custom_notes', '')).strip()

    parts = [
        f'Photorealistic interior design photograph of a {room}',
        style, theme, wcolor,
    ]
    if ceil:    parts.append(ceil)
    if floor:   parts.append(floor)
    if curtain: parts.append(curtain)
    if decor:   parts.append(decor)
    if plant:   parts.append(plant)
    parts += [
        light, budget,
        f'room contains: {objs}',
        'architectural digest editorial interior photography',
        'ultra-realistic 8K, professional DSLR, perfect exposure, sharp focus',
        'accurate colour reproduction, hyperrealistic materials and textures',
    ]

    if other:
        # Boost weight of custom notes to ensure AI attention
        parts.append(f"[[ MANDATORY USER REQUIREMENT: {other} ]]")

    prompt = ', '.join(p for p in parts if p)

    negative = (
        'cartoon, illustration, sketch, CGI render, blurry, low resolution, '
        'pixelated, watermark, text, people, humans, animals, '
        'distorted perspective, overexposed, bad lighting, ugly furniture, '
        'cluttered mess, dirty surfaces, chromatic aberration'
    )

    word_count = len(prompt.split())
    print(f'[Prompt] {word_count} words (~{int(word_count*1.3)} tokens) — OK with compel/SD2.1/SDXL')

    return prompt, negative