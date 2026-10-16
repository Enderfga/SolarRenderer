"""Biome / character / lighting tables for the Blender game-clip generator.

Asset refs are (pack_key, stem_regex, scale).  Units: KayKit units (a humanoid is ~2.4 tall,
1 unit ~= 0.75 m).  All packs are CC0 (KayKit by Kay Lousberg, Kenney, Quaternius, Poly Haven).
"""

PACKS = {
    "kk_city": "KayKit-City-Builder-Bits-1.0/",
    "kk_dun": "KayKit-Dungeon-Remastered-1.0/",
    "kk_hal": "KayKit-Halloween-Bits-1.0/",
    "kk_med": "KayKit-Medieval-Hexagon-Pack-1.0/",
    "kk_space": "KayKit-Space-Base-Bits-1.0/",
    "kk_furn": "KayKit-Furniture-Bits-1.0/",
    "kk_rest": "KayKit-Restaurant-Bits-1.0/",
    "kk_proto": "KayKit-Prototype-Bits-1.0/",
    "kk_adv": "KayKit-Character-Pack-Adventures-1.0/",
    "kk_skel": "KayKit-Character-Pack-Skeletons-1.0/",
    "kn_castle": "kenney_castle-kit/",
    "kn_comm": "kenney_city-kit-commercial",
    "kn_sub": "kenney_city-kit-suburban",
    "kn_town": "kenney_fantasy-town-kit",
    "kn_grave": "kenney_graveyard-kit",
    "kn_mdun": "kenney_mini-dungeon/",
    "kn_nature": "kenney_nature-kit/",
    "kn_space": "kenney_space-kit/",
    "kn_surv": "kenney_survival-kit/",
    "q_animals": "quat/ultimateanimatedanimals/",
    "q_monsters": "quat/ultimatemonsters/",
}
PACK_CREDIT = {
    "kk_city": "KayKit City Builder Bits", "kk_dun": "KayKit Dungeon Remastered", "kk_hal": "KayKit Halloween Bits",
    "kk_med": "KayKit Medieval Hexagon Pack", "kk_space": "KayKit Space Base Bits", "kk_furn": "KayKit Furniture Bits",
    "kk_rest": "KayKit Restaurant Bits", "kk_proto": "KayKit Prototype Bits", "kk_adv": "KayKit Adventurers",
    "kk_skel": "KayKit Skeletons", "kn_castle": "Kenney Castle Kit", "kn_comm": "Kenney City Kit (Commercial)",
    "kn_sub": "Kenney City Kit (Suburban)", "kn_town": "Kenney Fantasy Town Kit", "kn_grave": "Kenney Graveyard Kit",
    "kn_mdun": "Kenney Mini Dungeon", "kn_nature": "Kenney Nature Kit", "kn_space": "Kenney Space Kit",
    "kn_surv": "Kenney Survival Kit", "q_animals": "Quaternius Ultimate Animated Animals",
    "q_monsters": "Quaternius Ultimate Monsters",
}

# --------------------------------------------------------------------------------------------
# Characters.  anims: class -> candidate action names (filtered by what the rig has).
KK_ANIMS = {
    "walk": ["Walking_A", "Walking_B", "Walking_C"],
    "run": ["Running_A", "Running_B", "Running_C"],
    "idle": ["Idle", "Unarmed_Idle", "2H_Melee_Idle", "Idle_B", "Idle_Combat"],
    "attack": ["1H_Melee_Attack_Chop", "1H_Melee_Attack_Slice_Diagonal", "1H_Melee_Attack_Slice_Horizontal",
               "1H_Melee_Attack_Stab", "2H_Melee_Attack_Chop", "2H_Melee_Attack_Spin", "2H_Melee_Attack_Slice",
               "Unarmed_Melee_Attack_Punch_A", "Unarmed_Melee_Attack_Kick", "Dualwield_Melee_Attack_Chop"],
    "cast": ["Spellcasting", "Spellcast_Shoot", "Spellcast_Long", "Spellcast_Raise", "1H_Ranged_Shooting", "2H_Ranged_Shooting"],
    "interact": ["Interact", "PickUp", "Use_Item", "Throw"],
    "cheer": ["Cheer"],
    "block": ["Blocking", "Block"],
    "sit": ["Sit_Floor_Idle", "Sit_Chair_Idle"],
    "jump": ["Jump_Full_Long", "Jump_Full_Short"],
}
KN_ANIMS = {"walk": ["walk"], "run": ["sprint"], "idle": ["idle"], "attack": ["attack-melee-left", "attack-melee-right", "attack-kick-left", "attack-kick-right"],
            "interact": ["interact-left", "interact-right", "pick-up"], "cheer": ["emote-yes", "emote-no"], "jump": ["jump"]}
QA_ANIMS = {"walk": ["Walk"], "run": ["Gallop"], "idle": ["Idle", "Idle_2", "Idle_Headlow", "Idle_2_HeadLow", "Eating"],
            "attack": ["Attack_Headbutt", "Attack_Kick", "Attack"], "jump": ["Gallop_Jump"]}
QM_ANIMS = {"walk": ["Walk"], "run": ["Run"], "idle": ["Idle"], "attack": ["Punch", "Weapon", "Bite_Front", "Headbutt"],
            "cheer": ["Wave", "Yes", "Dance"], "jump": ["Jump"], "interact": ["Wave", "Yes", "No"]}

# name, pack, stem regex, kind, height target (units), anim table, extra
CHARACTERS = {
    "knight": ("kk_adv", "Knight", "kk", 1.0, KK_ANIMS, dict(weapons=[["1H_Sword", "Round_Shield"], ["1H_Sword", "Badge_Shield"], ["2H_Sword"], ["1H_Sword", "Spike_Shield"]], body="Knight_")),
    "barbarian": ("kk_adv", "Barbarian", "kk", 1.0, KK_ANIMS, dict(weapons=[["1H_Axe", "Barbarian_Round_Shield"], ["2H_Axe"], ["1H_Axe", "1H_Axe_Offhand"], ["Mug"]], body="Barbarian_")),
    "mage": ("kk_adv", "Mage", "kk", 1.0, KK_ANIMS, dict(weapons=[["2H_Staff"], ["1H_Wand", "Spellbook"], ["1H_Wand"]], body="Mage_")),
    "rogue": ("kk_adv", "Rogue", "kk", 1.0, KK_ANIMS, dict(weapons=[["Knife", "Knife_Offhand"], ["1H_Crossbow"], ["2H_Crossbow"]], body="Rogue_")),
    "hooded rogue": ("kk_adv", "Rogue_Hooded", "kk", 1.0, KK_ANIMS, dict(weapons=[["Knife", "Knife_Offhand"], ["1H_Crossbow"], ["2H_Crossbow"]], body="Rogue_")),
    "skeleton mage": ("kk_skel", "Skeleton_Mage", "kk", 1.0, KK_ANIMS, dict(weapons=[["Skeleton_Staff"], []], body="Skeleton_Mage_")),
    "skeleton minion": ("kk_skel", "Skeleton_Minion", "kk", 1.0, KK_ANIMS, dict(weapons=[["Skeleton_Blade"], ["Skeleton_Axe"], []], body="Skeleton_Minion_")),
    "skeleton rogue": ("kk_skel", "Skeleton_Rogue", "kk", 1.0, KK_ANIMS, dict(weapons=[["Skeleton_Crossbow"], ["Skeleton_Blade"]], body="Skeleton_Rogue_")),
    "skeleton warrior": ("kk_skel", "Skeleton_Warrior", "kk", 1.0, KK_ANIMS, dict(weapons=[["Skeleton_Blade", "Skeleton_Shield_Large_A"], ["Skeleton_Axe", "Skeleton_Shield_Small_A"], ["Skeleton_Blade"]], body="Skeleton_Warrior_")),
    "human adventurer": ("kn_mdun", "character-human", "kn", 1.15, KN_ANIMS, {}),
    "orc": ("kn_mdun", "character-orc", "kn", 1.2, KN_ANIMS, {}),
}
ANIMALS = {  # name: (stem, scale)
    "horse": ("Horse", 0.42), "white horse": ("Horse_White", 0.42), "deer": ("Deer", 0.38), "stag": ("Stag", 0.4),
    "fox": ("Fox", 0.24), "wolf": ("Wolf", 0.3), "husky": ("Husky", 0.26), "shiba inu": ("ShibaInu", 0.22),
    "cow": ("Cow", 0.36), "bull": ("Bull", 0.38), "donkey": ("Donkey", 0.36), "alpaca": ("Alpaca", 0.36),
}
for _n, (_s, _k) in ANIMALS.items():
    CHARACTERS[_n] = ("q_animals", _s, "qa", _k, QA_ANIMS, dict(animal=True))

HEROES = ["knight", "barbarian", "mage", "rogue", "hooded rogue", "human adventurer"]
FOES = ["skeleton mage", "skeleton minion", "skeleton rogue", "skeleton warrior", "orc"]

# --------------------------------------------------------------------------------------------
# Biomes.
A = lambda pack, rx, s=1.0, **kw: dict(pack=pack, rx=rx, s=s, **kw)

BIOMES = {
    "city street": dict(
        layout="grid", T=8.0, weight=1.0,
        road=[A("kk_city", "road_straight", 4), A("kk_city", "road_straight_crossing", 4)],
        junction=A("kk_city", "road_junction", 4), sidewalk=A("kk_city", "base", 4),
        buildings=[A("kk_city", "building_[A-H]_withoutBase", 4, base=True), A("kn_comm", "building-[a-n]", 7),
                   A("kn_comm", "building-skyscraper-[a-e]", 6.5), A("kn_sub", "building-type-[a-u]", 6)],
        edge=[A("kk_city", "streetlight", 4, lamp=0.95, w=3), A("kk_city", "bench|firehydrant|trash_[AB]|dumpster|box_[AB]", 4, w=4),
              A("kk_city", "bush", 4), A("kn_sub", "tree-(small|large)|planter", 6), A("kk_city", "trafficlight_[ABC]", 4, w=0.5)],
        vehicles=[A("kk_city", "car_(hatchback|police|sedan|stationwagon|taxi)", 4)],
        ground=(0.20, 0.30, 0.14), ground2=(0.16, 0.22, 0.10), path=None,
        chars=dict(hero=HEROES, npc=HEROES + FOES + ["husky", "shiba inu"]),
        tod={"morning": 1, "noon": 1, "afternoon": 1, "dusk": 1, "night": 1.2},
        words=("city street", "a paved city street lined with low-poly buildings, street lights and parked cars")),
    "dungeon": dict(
        layout="dungeon", T=4.0, weight=1.0, indoor=True,
        floors=[A("kk_dun", "floor_tile_large", 1), A("kk_dun", "floor_tile_large_rocks", 1), A("kk_dun", "floor_wood_large", 1),
                A("kk_dun", "floor_dirt_large", 1)],
        walls=[A("kk_dun", "wall(_[a-z_]+)?", 1, sizefilter=(3.8, 4.2, 0.0, 1.3, 3.5, 4.5))],
        pillar=[A("kk_dun", "pillar(_decorated)?|column", 1)],
        edge=[A("kk_dun", "barrel_(large|small)(_decorated)?|barrel_small_stack|box_(large|small|stacked)|box_small_decorated|crates_stacked|chest(_gold)?|keg(_decorated)?|shelf_(large|small)|shelves|table_[a-z_]+|chair|stool|bed_[a-z]+|rubble_(half|large)|sword_shield(_broken)?|coin_stack_large|trunk_[a-z_]+", 1, w=5),
              A("kk_dun", "torch_lit|candle_(lit|triple)|candle_thin_lit", 1, lamp=0.9, w=2),
              A("kn_mdun", "barrel|chest|rocks|stones|pot|wood-structure|banner|column", 3.6, w=2),
              A("kk_hal", "candle_triple|skull_candle|lantern_standing", 1, lamp=0.9, w=1), A("kk_hal", "bone_[ABC]|ribcage|skull|coffin", 1, w=1)],
        ground=(0.10, 0.09, 0.08), ground2=(0.07, 0.06, 0.05), path=None,
        chars=dict(hero=HEROES, npc=FOES + HEROES),
        tod={"indoor": 1},
        words=("dungeon", "a stone dungeon with torch-lit halls, barrels, crates and treasure chests")),
    "forest": dict(
        layout="free", weight=1.0, hw=(4.5, 7.0), lane=2.6,
        lining=[A("kn_nature", "tree_(blocks|cone|default|detailed|fat|oak|plateau|simple|small|tall|thin)(_dark)?|tree_pine(Default|Round|Tall)[A-F](_detailed)?", 4.5, jit=0.3),
                A("kn_nature", "rock_tall[A-J]|stone_tall[A-J]|cliff_block_rock|cliff_large_rock", 4.5, jit=0.3, w=0.3)],
        fill=[A("kn_nature", "tree_(blocks|cone|default|detailed|fat|oak|plateau|simple|tall|thin)(_dark)?|tree_pine(Default|Round|Tall)[A-F]", 4.5, jit=0.35)],
        edge=[A("kn_nature", "plant_bush(Large|Detailed|Small)?|flower_(purple|red|yellow)[ABC]|mushroom_(red|tan)(Group|Tall)?|stump_(old|round|square)\\w*|log(_large|_stack|_stackLarge)?|rock_small[A-I]|stone_small[A-I]|grass_large|grass_leafsLarge", 4.5, jit=0.25, w=6),
              A("kn_nature", "tent_(detailed|small)(Closed|Open)|campfire_(logs|stones|bricks)", 4.5, w=1, lamp=3.0, lampfilter="campfire"),
              A("kn_surv", "tent|tent-canvas|campfire-pit|barrel|box|bedroll|signpost|workbench|resource-wood|tree-log", 5, w=1, lamp=3.0, lampfilter="campfire")],
        scatter=[A("kn_nature", "grass(_large|_leafs|_leafsLarge)?|flower_(purple|red|yellow)[ABC]|plant_(bush|flatShort|flatTall)\\w*", 4.5, jit=0.3)],
        variants={"autumn": dict(lining_rx_sub=("(_dark)?", "_fall")), "dark pine": {}},
        ground=(0.22, 0.36, 0.12), ground2=(0.14, 0.26, 0.08), path=(0.36, 0.27, 0.17),
        chars=dict(hero=HEROES, npc=HEROES + FOES + ["deer", "stag", "fox", "wolf", "horse", "white horse"]),
        tod={"morning": 1.2, "noon": 1, "afternoon": 1, "dusk": 1, "night": 0.6},
        words=("forest", "a forest trail among low-poly trees, rocks, bushes and flowers")),
    "medieval village": dict(
        layout="free", weight=1.0, hw=(6.0, 8.5), lane=3.0,
        lining=[A("kk_med", "building_(home_A|home_B|tavern|blacksmith|church|market|barracks|lumbermill|watermill|windmill|well|tower_A|tower_B|archeryrange|mine)_(blue|green|red|yellow)", 6.5, jit=0.08),
                A("kn_town", "tree(-crooked|-high|-high-crooked|-high-round)?|hedge-large|stall(-green|-red)?|cart(-high)?", 4, w=0.5)],
        fill=[A("kk_med", "building_(castle|tower_A|tower_B|tower_base|church|windmill|home_B)_(blue|green|red|yellow)", 6.5),
              A("kn_castle", "tower-square|tower-hexagon-base|siege-(catapult|trebuchet|tower|ballista)|tree-large", 4.5),
              A("kk_med", "tree_single_[AB]|trees_[A-D]_(large|medium)|mountain_[ABC]_grass_trees", 6.5, w=0.5)],
        edge=[A("kn_town", "lantern", 4, lamp=0.9, w=2), A("kn_town", "cart|stall-bench|stall-stool|fence|fence-broken|hedge|rock-(large|small|wide)|banner-(green|red)|fountain-round|tree", 4, w=6),
              A("kn_surv", "barrel|box|resource-planks|resource-wood|signpost", 5, w=2)],
        scatter=[A("kn_nature", "grass(_large)?|flower_(purple|red|yellow)[ABC]", 4.5, jit=0.3)],
        ground=(0.26, 0.40, 0.15), ground2=(0.20, 0.32, 0.10), path=(0.46, 0.42, 0.36),
        chars=dict(hero=HEROES, npc=HEROES + FOES + ["horse", "white horse", "cow", "donkey", "alpaca", "husky"]),
        tod={"morning": 1, "noon": 1, "afternoon": 1, "dusk": 1, "night": 0.8},
        words=("medieval village", "a medieval village road with houses, a tavern, market stalls and lanterns")),
    "sci-fi base": dict(
        layout="free", weight=1.0, hw=(6.5, 9.0), lane=3.0,
        lining=[A("kk_space", "basemodule_[A-E]|basemodule_garage|cargodepot_[ABC]|structure_(low|tall)|drill_structure|lander_[AB]|windturbine_(low|tall)|solarpanel|containers_[A-D]", 5, jit=0.05),
                A("kn_space", "hangar_(large[AB]|round[AB]|roundGlass|small[AB])|machine_(generator|generatorLarge|wireless)|satelliteDish(_detailed|_large)?|turret_(double|single)|rocket_baseA", 5)],
        fill=[A("kk_space", "rock_[AB]|rocks_[AB]|windturbine_tall|structure_tall|lander_base", 5, jit=0.2),
              A("kn_space", "rock_large[AB]|meteor(_detailed)?|rock_crystalsLarge[AB]|satelliteDish_large|hangar_large[AB]", 5, jit=0.2)],
        edge=[A("kk_space", "cargo_[AB](_packed|_stacked)?|spacetruck|rocks_[AB]", 5, w=4), A("kk_space", "lights", 5, lamp=0.9, w=2),
              A("kn_space", "barrel(s)?|machine_barrel(Large)?|rover|rocks_small[AB]|rock_crystals|pipe_(straight|ring|supportHigh)|craft_(speeder[A-D]|racer|miner)", 5, w=5)],
        ground=(0.45, 0.24, 0.13), ground2=(0.33, 0.17, 0.09), path=(0.30, 0.30, 0.32),
        sky_tint=[(1.0, 0.75, 0.6), (0.8, 0.9, 1.1), (1.0, 1.0, 1.0)],
        chars=dict(hero=HEROES, npc=HEROES + FOES),
        tod={"morning": 1, "noon": 1, "afternoon": 1, "dusk": 1, "night": 0.8},
        words=("sci-fi base", "a sci-fi outpost on a rocky planet with base modules, cargo, landers and antennas")),
    "graveyard": dict(
        layout="free", weight=0.8, hw=(5.0, 7.0), lane=2.8,
        lining=[A("kk_hal", "crypt|fence(_broken)?|tree_dead_(large|medium)(_decorated)?|tree_pine_(orange|yellow)_(large|medium)|arch|shrine(_candles)?", 1.0, jit=0.1),
                A("kn_grave", "crypt(-a|-b|-large)?|pine(-crooked|-fall|-fall-crooked)?|stone-wall(-damaged)?|iron-fence(-damaged)?|pillar-(large|obelisk)", 4, jit=0.1)],
        fill=[A("kk_hal", "tree_dead_(large|medium)|tree_pine_(orange|yellow)_large|crypt", 1.0, jit=0.2), A("kn_grave", "pine(-crooked|-fall)?", 4, jit=0.2)],
        edge=[A("kk_hal", "grave_[AB]|grave_A_destroyed|gravestone|gravemarker_[AB]|pumpkin_(orange|yellow)(_small|_jackolantern)?|skull|bone_[ABC]|ribcage|bench(_decorated)?|coffin(_decorated)?|tree_dead_small|post_skull", 1.0, w=8),
              A("kk_hal", "lantern_standing|post_lantern|candle_triple|skull_candle|plaque_candles", 1.0, lamp=0.85, w=3),
              A("kn_grave", "gravestone-(bevel|broken|cross|cross-large|decorative|roof|round|wide)|grave|pumpkin(-carved|-tall-carved)?|urn-(round|square)|debris|shovel-dirt|hay-bale|bench(-damaged)?|cross(-wood)?", 4, w=6),
              A("kn_grave", "lightpost-(single|double)|lantern-candle|fire-basket|candle-multiple", 4, lamp=0.9, w=2)],
        ground=(0.13, 0.17, 0.09), ground2=(0.10, 0.11, 0.07), path=(0.24, 0.20, 0.15),
        chars=dict(hero=HEROES, npc=FOES + HEROES + ["wolf", "fox"]),
        tod={"dusk": 1.0, "night": 1.6, "afternoon": 0.4, "morning": 0.3},
        words=("graveyard", "a spooky graveyard with crypts, gravestones, dead trees, pumpkins and lanterns")),
}

# --------------------------------------------------------------------------------------------
# Lighting presets: time of day -> sun elevation range (deg), sun energy, sun colour, HDRIs, sky strength.
HDRI = {
    "clear": ["kloofendal_43d_clear_puresky", "syferfontein_0d_clear_puresky", "table_mountain_1_puresky"],
    "cloudy": ["kloofendal_48d_partly_cloudy_puresky", "kloppenheim_06_puresky", "rustig_koppie_puresky", "qwantani_moon_noon_puresky"],
    "overcast": ["overcast_soil_puresky"],
    "sunset": ["belfast_sunset_puresky", "qwantani_dusk_2_puresky"],
    "night": ["qwantani_night_puresky"],
}
TOD = {
    "morning": dict(elev=(8, 22), energy=(2.2, 3.2), color=(1.0, 0.86, 0.7), sky=["clear", "cloudy"], sky_strength=(0.7, 1.0), lamps=False),
    "noon": dict(elev=(50, 75), energy=(3.5, 5.0), color=(1.0, 0.98, 0.95), sky=["clear", "cloudy", "overcast"], sky_strength=(0.9, 1.2), lamps=False),
    "afternoon": dict(elev=(25, 45), energy=(3.0, 4.2), color=(1.0, 0.93, 0.82), sky=["clear", "cloudy", "overcast"], sky_strength=(0.8, 1.1), lamps=False),
    "dusk": dict(elev=(2, 9), energy=(1.5, 2.6), color=(1.0, 0.55, 0.3), sky=["sunset"], sky_strength=(0.6, 1.0), lamps=True),
    "night": dict(elev=(25, 60), energy=(0.15, 0.35), color=(0.55, 0.65, 1.0), sky=["night"], sky_strength=(0.12, 0.3), lamps=True),
    "indoor": dict(elev=(55, 80), energy=(0.05, 0.25), color=(0.6, 0.7, 1.0), sky=["night"], sky_strength=(0.15, 0.4), lamps=True),
}
