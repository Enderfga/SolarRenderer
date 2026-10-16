"""Poly Haven (CC0) texture sets for the 'realistic' render style.  A clip draws one set per surface class."""
CLASSES = {
    "foliage": ["forest_leaves_02", "leafy_grass", "sparse_grass"],
    "wood": ["wood_planks", "bark_brown_02", "brown_planks_03", "wood_table_001"],
    "stone_wall": ["rock_wall_08", "castle_wall_slates", "red_brick_03", "rock_face", "rough_plasterbrick_05"],
    "stone_floor": ["stone_tiles_02", "cobblestone_floor_01", "concrete_floor_02", "gravel_concrete", "asphalt_02"],
    "painted": ["painted_plaster_wall", "beige_wall_001", "plastered_wall", "painted_concrete"],
    "fabric": ["denim_fabric", "fabric_leather_02", "leather_red_03"],
    "metal": ["metal_plate_02", "rusty_painted_metal"],
}
GROUND = {  # biome -> ground plane / path ribbon sets
    "city street": (["leafy_grass", "sparse_grass"], ["gravel_concrete"]),
    "dungeon": (["stone_tiles_02", "cobblestone_floor_01"], ["cobblestone_floor_01"]),
    "forest": (["forrest_ground_01", "brown_mud_leaves_01"], ["dirt", "grass_path_2"]),
    "medieval village": (["aerial_grass_rock", "sparse_grass"], ["cobblestone_floor_01", "dirt"]),
    "graveyard": (["burned_ground_01", "forrest_ground_01"], ["dirt", "gravel_concrete"]),
    "_default": (["dry_ground_01", "rocky_terrain_02", "coast_sand_01"], ["dirt", "sandstone_cracks"]),
}
ALL_SETS = sorted({s for v in CLASSES.values() for s in v} | {s for g, p in GROUND.values() for s in g + p})
