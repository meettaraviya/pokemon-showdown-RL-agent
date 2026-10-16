"""Feature extraction for the isolated RL training package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from pathlib import Path

import numpy as np

from ..ou_selfplay_client import BattleState, PlayerId
from .custom_types import BattleFeatures

STATUS_ORDER: tuple[str, ...] = ("healthy", "brn", "psn", "tox", "par", "slp", "frz")
TYPE_ORDER: tuple[str, ...] = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
)

HP_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0)
REQUEST_KIND_ORDER: tuple[str, ...] = ("none", "team_preview", "force_switch", "active", "wait", "unknown")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RL_DATA_ROOT = Path(__file__).resolve().parent / "data"
VOCAB_CACHE_PATH = RL_DATA_ROOT / "entity_vocab_cache.json"
VOCAB_CACHE_VERSION = 2
SV_OU_SPECIES_PATH = RL_DATA_ROOT / "sv_ou_species.txt"
SV_OU_ITEMS_PATH = RL_DATA_ROOT / "sv_ou_items.txt"
SV_OU_MOVES_PATH = RL_DATA_ROOT / "sv_ou_moves.txt"
SV_OU_ABILITIES_PATH = RL_DATA_ROOT / "sv_ou_abilities.txt"


def _normalize_vocab_token(value: object) -> str:
    """_normalize_vocab_token utility used by the RL self-play workflow."""
    if value is None:
        return "<unk>"
    text = str(value).strip().lower()
    if not text:
        return "<unk>"
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text or "<unk>"


class GlobalVocabulary:
    """GlobalVocabulary helper type/container used by the RL self-play tooling."""
    def __init__(self, name: str, unknown_token: str = "<unk>") -> None:
        self.name = name
        self.unknown_token = unknown_token
        self._index = {unknown_token: 0}
        self._items = [unknown_token]

    def register(self, item: object) -> int:
        token = _normalize_vocab_token(item)
        if token not in self._index:
            self._index[token] = len(self._items)
            self._items.append(token)
        return self._index[token]

    def one_hot(self, item: object) -> np.ndarray:
        token = _normalize_vocab_token(item)
        index = self._index.get(token, self._index[self.unknown_token])
        vector = np.zeros(self.size(), dtype=np.float32)
        vector[index] = 1.0
        return vector

    def size(self) -> int:
        return len(self._items)


def _load_table_keys(export_name: str, data_file: str) -> list[str]:
    """_load_table_keys utility used by the RL self-play workflow."""
    path = DATA_ROOT / data_file
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    marker = f"export const {export_name}"
    marker_index = text.find(marker)
    if marker_index < 0:
        return []
    table_start = text.find("{", marker_index)
    if table_start < 0:
        return []

    keys: list[str] = []
    depth = 0
    has_opened = False
    for line in text[table_start:].splitlines():
        if depth == 1:
            match = re.match(r"^\s*(?:\"([^\"]+)\"|([A-Za-z0-9_]+))\s*:\s*\{", line)
            if match:
                key = match.group(1) or match.group(2)
                if key:
                    keys.append(key)

        depth += line.count("{") - line.count("}")
        if depth > 0:
            has_opened = True
        elif has_opened:
            break

    return keys


def _vocab_source_files() -> dict[str, str]:
    """_vocab_source_files utility used by the RL self-play workflow."""
    return {
        "species": str(SV_OU_SPECIES_PATH),
        "items": str(SV_OU_ITEMS_PATH),
        "abilities": str(SV_OU_ABILITIES_PATH),
        "moves": str(SV_OU_MOVES_PATH),
    }


def _source_mtimes() -> dict[str, int]:
    """_source_mtimes utility used by the RL self-play workflow."""
    mtimes: dict[str, int] = {}
    for filename in _vocab_source_files().values():
        path = Path(filename)
        if not path.is_absolute():
            path = DATA_ROOT / filename
        mtimes[filename] = path.stat().st_mtime_ns if path.exists() else -1
    return mtimes


def _read_vocab_text_file(path: Path) -> list[str]:
    """_read_vocab_text_file utility used by the RL self-play workflow."""
    if not path.exists():
        return []
    tokens: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


def _load_cached_vocab_tokens() -> dict[str, list[str]] | None:
    """_load_cached_vocab_tokens utility used by the RL self-play workflow."""
    if not VOCAB_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(VOCAB_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if payload.get("version") != VOCAB_CACHE_VERSION:
        return None
    if payload.get("source_mtimes") != _source_mtimes():
        return None

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None

    out: dict[str, list[str]] = {}
    for key in ("species", "items", "abilities", "moves"):
        values = tokens.get(key)
        if not isinstance(values, list):
            return None
        out[key] = [str(value) for value in values]
    return out


def _write_vocab_cache(tokens: dict[str, list[str]]) -> None:
    """_write_vocab_cache utility used by the RL self-play workflow."""
    VOCAB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VOCAB_CACHE_VERSION,
        "source_mtimes": _source_mtimes(),
        "tokens": tokens,
    }
    VOCAB_CACHE_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _load_global_vocab_tokens() -> dict[str, list[str]]:
    """_load_global_vocab_tokens utility used by the RL self-play workflow."""
    cached = _load_cached_vocab_tokens()
    if cached is not None:
        return cached

    species_tokens = _read_vocab_text_file(SV_OU_SPECIES_PATH)
    item_tokens = _read_vocab_text_file(SV_OU_ITEMS_PATH)
    ability_tokens = _read_vocab_text_file(SV_OU_ABILITIES_PATH)
    move_tokens = _read_vocab_text_file(SV_OU_MOVES_PATH)

    tokens = {
        "species": species_tokens if species_tokens else _load_table_keys("Pokedex", "pokedex.ts"),
        "items": item_tokens if item_tokens else _load_table_keys("Items", "items.ts"),
        "abilities": ability_tokens if ability_tokens else _load_table_keys("Abilities", "abilities.ts"),
        "moves": move_tokens if move_tokens else _load_table_keys("Moves", "moves.ts"),
    }
    _write_vocab_cache(tokens)
    return tokens


SPECIES_VOCAB = GlobalVocabulary("species")
ITEM_VOCAB = GlobalVocabulary("item")
ABILITY_VOCAB = GlobalVocabulary("ability")
MOVE_VOCAB = GlobalVocabulary("move")

GLOBAL_VOCAB_TOKENS = _load_global_vocab_tokens()

for name in GLOBAL_VOCAB_TOKENS["species"]:
    SPECIES_VOCAB.register(name)
for name in GLOBAL_VOCAB_TOKENS["items"]:
    ITEM_VOCAB.register(name)
for name in GLOBAL_VOCAB_TOKENS["abilities"]:
    ABILITY_VOCAB.register(name)
for name in GLOBAL_VOCAB_TOKENS["moves"]:
    MOVE_VOCAB.register(name)

SPECIES_EMBED_DIM = SPECIES_VOCAB.size()
ITEM_EMBED_DIM = ITEM_VOCAB.size()
ABILITY_EMBED_DIM = ABILITY_VOCAB.size()
BASE_ABILITY_EMBED_DIM = ABILITY_VOCAB.size()
TERA_EMBED_DIM = 6
MOVESET_EMBED_DIM = 4 * MOVE_VOCAB.size()

MON_TERA_PROFILE_START = (
    SPECIES_EMBED_DIM
    + ITEM_EMBED_DIM
    + ABILITY_EMBED_DIM
    + BASE_ABILITY_EMBED_DIM
    + TERA_EMBED_DIM
    + MOVESET_EMBED_DIM
    + 6
    + 1
    + len(HP_BUCKETS)
    + len(STATUS_ORDER)
    + 1
    + 6
    + 7
    + 6
)
MON_HP_FRAC_INDEX = (
    SPECIES_EMBED_DIM
    + ITEM_EMBED_DIM
    + ABILITY_EMBED_DIM
    + BASE_ABILITY_EMBED_DIM
    + TERA_EMBED_DIM
    + MOVESET_EMBED_DIM
    + 6
)
MON_FAINTED_FLAG_INDEX = (
    SPECIES_EMBED_DIM
    + ITEM_EMBED_DIM
    + ABILITY_EMBED_DIM
    + BASE_ABILITY_EMBED_DIM
    + TERA_EMBED_DIM
    + MOVESET_EMBED_DIM
    + 6
    + 1
    + len(HP_BUCKETS)
    + len(STATUS_ORDER)
    + 1
    + 6
    + 1
)
MON_FEATURE_DIM = MON_TERA_PROFILE_START + len(TYPE_ORDER) + 1 + len(TYPE_ORDER)

TYPE_ATTACK_CHART: dict[str, tuple[float, ...]] = {
    "normal": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0.5, 1),
    "fire": (1, 0.5, 0.5, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0.5, 1, 0.5, 1, 2, 1),
    "water": (1, 2, 0.5, 1, 0.5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0.5, 1, 1, 1),
    "electric": (1, 1, 2, 0.5, 0.5, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0.5, 1, 1, 1),
    "grass": (1, 0.5, 2, 1, 0.5, 1, 1, 0.5, 2, 0.5, 1, 0.5, 2, 1, 0.5, 1, 0.5, 1),
    "ice": (1, 0.5, 0.5, 1, 2, 0.5, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 0.5, 1),
    "fighting": (2, 1, 1, 1, 1, 2, 1, 0.5, 1, 0.5, 0.5, 0.5, 2, 0, 1, 2, 2, 0.5),
    "poison": (1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 0, 2),
    "ground": (1, 2, 1, 2, 0.5, 1, 1, 2, 1, 0, 1, 0.5, 2, 1, 1, 1, 2, 1),
    "flying": (1, 1, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 0.5, 1),
    "psychic": (1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0.5, 1, 1, 1, 1, 0, 0.5, 1),
    "bug": (1, 0.5, 1, 1, 2, 1, 0.5, 0.5, 1, 0.5, 2, 1, 1, 0.5, 1, 2, 0.5, 0.5),
    "rock": (1, 2, 1, 1, 1, 2, 0.5, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 0.5, 1),
    "ghost": (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0.5, 1, 1),
    "dragon": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0.5, 0),
    "dark": (1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1, 0.5),
    "steel": (1, 0.5, 0.5, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0.5, 2),
    "fairy": (1, 0.5, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 1, 2, 2, 0.5, 1),
}


def _request_kind(request: Mapping[str, object] | None) -> str:
    """_request_kind utility used by the RL self-play workflow."""
    if request is None:
        return "none"
    if request.get("teamPreview"):
        return "team_preview"
    if request.get("forceSwitch"):
        return "force_switch"
    if request.get("active"):
        return "active"
    if request.get("wait"):
        return "wait"
    return "unknown"


def _move_slot_vector(moves: Sequence[str], vocab: GlobalVocabulary) -> np.ndarray:
    """_move_slot_vector utility used by the RL self-play workflow."""
    vector = np.zeros(4 * vocab.size(), dtype=np.float32)
    for slot_index, move in enumerate(moves[:4]):
        block = vocab.one_hot(move)
        start = slot_index * vocab.size()
        vector[start : start + vocab.size()] = block
    return vector


def _safe_stats_map(pokemon: Mapping[str, object]) -> Mapping[str, object]:
    """_safe_stats_map utility used by the RL self-play workflow."""
    stats = pokemon.get("stats")
    return stats if isinstance(stats, Mapping) else {}


def _parse_condition(condition: str) -> tuple[float, str]:
    """_parse_condition utility used by the RL self-play workflow."""
    if not condition:
        return 0.0, "healthy"
    if condition.endswith(" fnt") or condition == "0 fnt":
        return 0.0, "healthy"
    tokens = condition.split()
    hp_token = tokens[0] if tokens else ""
    status_token = "healthy"
    if len(tokens) >= 2 and tokens[1].lower() in STATUS_ORDER:
        status_token = tokens[1].lower()
    if "/" in hp_token:
        hp_now_text, hp_max_text = hp_token.split("/", 1)
        try:
            hp_now = float(hp_now_text)
            hp_max = float(hp_max_text)
            if hp_max > 0:
                hp_frac = max(0.0, min(1.0, hp_now / hp_max))
            else:
                hp_frac = 0.0
        except ValueError:
            hp_frac = 0.0
    else:
        hp_frac = 1.0
    return hp_frac, status_token


def _hp_bucket_vector(hp_frac: float) -> np.ndarray:
    """_hp_bucket_vector utility used by the RL self-play workflow."""
    buckets = np.zeros(len(HP_BUCKETS), dtype=np.float32)
    if hp_frac <= 0.0:
        buckets[0] = 1.0
        return buckets
    if hp_frac <= 0.25:
        buckets[1] = 1.0
    elif hp_frac <= 0.5:
        buckets[2] = 1.0
    elif hp_frac <= 0.75:
        buckets[3] = 1.0
    else:
        buckets[4] = 1.0
    return buckets


def _status_vector(status: str) -> np.ndarray:
    """_status_vector utility used by the RL self-play workflow."""
    vector = np.zeros(len(STATUS_ORDER), dtype=np.float32)
    if status in STATUS_ORDER:
        vector[STATUS_ORDER.index(status)] = 1.0
    else:
        vector[0] = 1.0
    return vector


def _coerce_sequence(value: object) -> list[object]:
    """_coerce_sequence utility used by the RL self-play workflow."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _extract_species(details: str, ident: str) -> str:
    """_extract_species utility used by the RL self-play workflow."""
    if details:
        return details.split(",", 1)[0].strip().lower()
    if ident:
        return ident.split(":", 1)[-1].strip().lower()
    return "unknown"


def _normalize_stat(value: object) -> float:
    """_normalize_stat utility used by the RL self-play workflow."""
    try:
        stat = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, stat / 714.0))


def _extract_active_stages(request: Mapping[str, object], slot_index: int) -> np.ndarray:
    """_extract_active_stages utility used by the RL self-play workflow."""
    stages = np.zeros(6, dtype=np.float32)
    active = _coerce_sequence(request.get("active"))
    if slot_index >= len(active):
        return stages
    entry = active[slot_index]
    if not isinstance(entry, Mapping):
        return stages
    boosts = entry.get("boosts")
    if not isinstance(boosts, Mapping):
        return stages
    order = ("atk", "def", "spa", "spd", "spe", "accuracy")
    for i, key in enumerate(order):
        raw = boosts.get(key)
        if raw is None:
            continue
        try:
            stages[i] = float(int(raw)) / 6.0
        except (TypeError, ValueError):
            stages[i] = 0.0
    return np.clip(stages, -1.0, 1.0)


def _move_profile_from_side_entry(pokemon: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """_move_profile_from_side_entry utility used by the RL self-play workflow."""
    moves = [_normalize_vocab_token(move) for move in _coerce_sequence(pokemon.get("moves"))]
    if not moves:
        return np.zeros(MOVESET_EMBED_DIM, dtype=np.float32), np.zeros(6, dtype=np.float32)

    move_embed = _move_slot_vector(moves, MOVE_VOCAB)
    has_pivot = 0.0
    has_recovery = 0.0
    has_hazard_set = 0.0
    has_hazard_clear = 0.0
    has_setup = 0.0
    for move in moves:
        if move in {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "shedtail"}:
            has_pivot = 1.0
        if move in {"recover", "roost", "slackoff", "softboiled", "wish", "moonlight", "morningsun", "synthesis"}:
            has_recovery = 1.0
        if move in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}:
            has_hazard_set = 1.0
        if move in {"rapidspin", "defog", "mortalspin", "tidyup"}:
            has_hazard_clear = 1.0
        if move in {"swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "quiverdance"}:
            has_setup = 1.0
    move_profile = np.array(
        [
            min(1.0, len(moves) / 4.0),
            has_pivot,
            has_recovery,
            has_hazard_set,
            has_hazard_clear,
            has_setup,
        ],
        dtype=np.float32,
    )
    return move_embed, move_profile


def _tera_profile(tera_type: str) -> np.ndarray:
    """_tera_profile utility used by the RL self-play workflow."""
    tera = tera_type.strip().lower()
    if not tera or tera not in TYPE_ATTACK_CHART:
        return np.zeros(len(TYPE_ORDER), dtype=np.float32)
    return np.array(TYPE_ATTACK_CHART[tera], dtype=np.float32)


def _tera_onehot(tera_type: str) -> np.ndarray:
    """_tera_onehot utility used by the RL self-play workflow."""
    vector = np.zeros(len(TYPE_ORDER) + 1, dtype=np.float32)
    tera = tera_type.strip().lower()
    if tera in TYPE_ORDER:
        vector[TYPE_ORDER.index(tera)] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def _pokemon_token_features(
    pokemon: Mapping[str, object],
    *,
    side_request: Mapping[str, object],
    slot_index: int,
) -> np.ndarray:
    """_pokemon_token_features utility used by the RL self-play workflow."""
    details = str(pokemon.get("details", ""))
    ident = str(pokemon.get("ident", ""))
    species = _extract_species(details, ident)
    item = str(pokemon.get("item", "")).strip().lower()
    ability = str(pokemon.get("ability", "")).strip().lower()
    base_ability = str(pokemon.get("baseAbility", "")).strip().lower()
    tera_type = str(pokemon.get("teraType", "")).strip().lower()

    species_embed = SPECIES_VOCAB.one_hot(species)
    item_embed = ITEM_VOCAB.one_hot(item)
    ability_embed = ABILITY_VOCAB.one_hot(ability)
    base_ability_embed = ABILITY_VOCAB.one_hot(base_ability)
    tera_embed = np.zeros(TERA_EMBED_DIM, dtype=np.float32)
    if tera_type:
        tera_embed[0] = 1.0

    stats_map = _safe_stats_map(pokemon)
    stats = np.array(
        [
            _normalize_stat(stats_map.get("hp")),
            _normalize_stat(stats_map.get("atk")),
            _normalize_stat(stats_map.get("def")),
            _normalize_stat(stats_map.get("spa")),
            _normalize_stat(stats_map.get("spd")),
            _normalize_stat(stats_map.get("spe")),
        ],
        dtype=np.float32,
    )

    hp_frac, status = _parse_condition(str(pokemon.get("condition", "")))
    hp_bucket = _hp_bucket_vector(hp_frac)
    status_vector = _status_vector(status)
    is_statused = np.array([0.0 if status == "healthy" else 1.0], dtype=np.float32)

    stages = _extract_active_stages(side_request, slot_index)

    fainted = 1.0 if " fnt" in str(pokemon.get("condition", "")) or bool(pokemon.get("fainted")) else 0.0
    flags = np.array(
        [
            1.0 if bool(pokemon.get("active")) else 0.0,
            fainted,
            1.0 if bool(pokemon.get("reviving")) else 0.0,
            1.0 if bool(pokemon.get("terastallized")) else 0.0,
            1.0 if bool(pokemon.get("commanding")) else 0.0,
            0.0 if species == "unknown" else 1.0,
            0.0 if not item else 1.0,
        ],
        dtype=np.float32,
    )

    moveset_embed, move_profile = _move_profile_from_side_entry(pokemon)
    tera_onehot = _tera_onehot(tera_type)
    tera_profile = _tera_profile(tera_type)

    return np.concatenate(
        [
            species_embed,
            item_embed,
            ability_embed,
            base_ability_embed,
            tera_embed,
            moveset_embed,
            stats,
            np.array([hp_frac], dtype=np.float32),
            hp_bucket,
            status_vector,
            is_statused,
            stages,
            flags,
            move_profile,
            tera_onehot,
            tera_profile,
        ],
        dtype=np.float32,
    )


def _extract_side_pokemon(request: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    """_extract_side_pokemon utility used by the RL self-play workflow."""
    if not isinstance(request, Mapping):
        return []
    side = request.get("side")
    if not isinstance(side, Mapping):
        return []
    entries = _coerce_sequence(side.get("pokemon"))
    out: list[Mapping[str, object]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            out.append(entry)
    return out


def _build_party_features(state: BattleState, player: PlayerId) -> np.ndarray:
    """_build_party_features utility used by the RL self-play workflow."""
    own_request = state.requests.get(player)
    opp_player: PlayerId = "p2" if player == "p1" else "p1"
    opp_request = state.requests.get(opp_player)

    own_party = _extract_side_pokemon(own_request)
    opp_party = _extract_side_pokemon(opp_request)

    feature_rows: list[np.ndarray] = []
    for slot_index in range(6):
        if slot_index < len(own_party):
            token = _pokemon_token_features(own_party[slot_index], side_request=own_request or {}, slot_index=slot_index)
        else:
            token = np.zeros(MON_FEATURE_DIM, dtype=np.float32)
        feature_rows.append(token)

    for slot_index in range(6):
        if slot_index < len(opp_party):
            token = _pokemon_token_features(opp_party[slot_index], side_request=opp_request or {}, slot_index=slot_index)
        else:
            token = np.zeros(MON_FEATURE_DIM, dtype=np.float32)
        feature_rows.append(token)

    return np.stack(feature_rows, axis=0)


def _request_kind_onehot(kind: str) -> np.ndarray:
    """_request_kind_onehot utility used by the RL self-play workflow."""
    vector = np.zeros(len(REQUEST_KIND_ORDER), dtype=np.float32)
    if kind in REQUEST_KIND_ORDER:
        vector[REQUEST_KIND_ORDER.index(kind)] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def _request_mechanics_features(request: Mapping[str, object] | None) -> np.ndarray:
    """_request_mechanics_features utility used by the RL self-play workflow."""
    vector = np.zeros(16, dtype=np.float32)
    if not isinstance(request, Mapping):
        return vector

    if request.get("teamPreview"):
        vector[0] = 1.0
    if request.get("forceSwitch"):
        vector[1] = 1.0
    if request.get("wait"):
        vector[2] = 1.0

    active = _coerce_sequence(request.get("active"))
    for entry in active:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("canMegaEvo"):
            vector[3] = 1.0
        if entry.get("canZMove"):
            vector[4] = 1.0
        if entry.get("canDynamax"):
            vector[5] = 1.0
        if entry.get("canTerastallize"):
            vector[6] = 1.0
        if entry.get("mustSwitch"):
            vector[7] = 1.0
        if entry.get("trapped"):
            vector[8] = 1.0
        if entry.get("choiceType"):
            vector[9] = 1.0
        if entry.get("maxMoves"):
            vector[10] = 1.0
        if entry.get("canUseMove"):
            vector[11] = 1.0
        if _coerce_sequence(entry.get("moves")):
            vector[12] = 1.0
        choices = entry.get("choices")
        if isinstance(choices, Mapping) and choices:
            vector[13] = 1.0
        if entry.get("megaEvo"):
            vector[14] = 1.0
        if entry.get("zmove"):
            vector[15] = 1.0
    return vector


def _global_condition_features(request: Mapping[str, object] | None) -> np.ndarray:
    """_global_condition_features utility used by the RL self-play workflow."""
    weather_onehot = np.zeros(6, dtype=np.float32)
    terrain_onehot = np.zeros(5, dtype=np.float32)
    side_effects = np.zeros(12, dtype=np.float32)
    timing = np.zeros(8, dtype=np.float32)

    if isinstance(request, Mapping):
        weather = request.get("weather")
        if isinstance(weather, str):
            key = weather.lower()
            mapping = {"none": 0, "raindance": 1, "sunnyday": 2, "sandstorm": 3, "snow": 4, "hail": 5}
            if key in mapping:
                weather_onehot[mapping[key]] = 1.0

    mechanics = _request_mechanics_features(request)
    return np.concatenate([weather_onehot, terrain_onehot, side_effects, timing, mechanics], dtype=np.float32)


def build_battle_features(state: BattleState, player: PlayerId) -> BattleFeatures:
    """build_battle_features utility used by the RL self-play workflow."""
    request = state.request_for(player)
    legal_actions = tuple(state.possible_actions(player))
    request_kind = _request_kind(request)
    request_kind_vector = _request_kind_onehot(request_kind)

    party_features = _build_party_features(state, player)
    own_party = party_features[:6]
    opp_party = party_features[6:]
    own_hp_mean = float(np.mean(own_party[:, MON_HP_FRAC_INDEX]))
    opp_hp_mean = float(np.mean(opp_party[:, MON_HP_FRAC_INDEX]))
    own_fainted = float(np.mean(own_party[:, MON_FAINTED_FLAG_INDEX]))
    opp_fainted = float(np.mean(opp_party[:, MON_FAINTED_FLAG_INDEX]))

    pending_flags = np.array(
        [
            1.0 if "p1" in state.pending_players else 0.0,
            1.0 if "p2" in state.pending_players else 0.0,
            float(len(state.pending_players)) / 2.0,
        ],
        dtype=np.float32,
    )

    global_features = _global_condition_features(request)

    scalar_features = np.concatenate(
        [
            np.array(
                [
                    float(state.turns) / 100.0,
                    1.0 if state.done else 0.0,
                    1.0 if state.winner == player else 0.0,
                    1.0 if state.winner not in {None, player, "tie"} else 0.0,
                    float(len(state.errors.get(player, []))) / 4.0,
                    float(len(legal_actions)) / 32.0,
                    own_hp_mean,
                    opp_hp_mean,
                    own_fainted,
                    opp_fainted,
                ],
                dtype=np.float32,
            ),
            pending_flags,
            request_kind_vector,
            global_features,
        ],
        dtype=np.float32,
    )

    legal_action_mask = np.ones(len(legal_actions), dtype=np.bool_)
    return BattleFeatures(
        player_id=player,
        turn_index=state.turns,
        done=state.done,
        winner=state.winner,
        request_kind=request_kind,
        pending_players=tuple(state.pending_players),
        request=request,
        scalar_features=scalar_features,
        party_features=party_features,
        legal_actions=legal_actions,
        legal_action_mask=legal_action_mask,
    )


def heuristic_position_score(features: BattleFeatures) -> float:
    """Dense proxy for board quality based on HP and fainted share."""

    party = features.party_features
    if party.shape[0] < 12:
        return 0.0
    own = party[:6]
    opp = party[6:12]
    own_hp = float(np.mean(own[:, MON_HP_FRAC_INDEX]))
    opp_hp = float(np.mean(opp[:, MON_HP_FRAC_INDEX]))
    own_fainted = float(np.mean(own[:, MON_FAINTED_FLAG_INDEX]))
    opp_fainted = float(np.mean(opp[:, MON_FAINTED_FLAG_INDEX]))
    return 0.6 * (own_hp - opp_hp) + 0.4 * (opp_fainted - own_fainted)
