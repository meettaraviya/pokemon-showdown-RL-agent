"""Generate SV OU entity vocabularies from Pokemon Showdown sources.

This script derives species, abilities, moves, and items from Showdown's own
runtime data/rules for a target format. It also derives a non-battle item list
without using external team corpora.
"""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import json
from pathlib import Path
import subprocess
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[3]
RL_DATA_DIR = Path(__file__).resolve().parent / "data"


def _run_node_vocab_builder(repo_root: Path, format_id: str) -> dict:
    """_run_node_vocab_builder utility used by the RL self-play workflow."""
    js_code = textwrap.dedent(
        """
        const fs = require('fs');
        const path = require('path');

        const repoRoot = process.argv[1];
        const formatId = process.argv[2];
        const {Dex, toID} = require(path.join(repoRoot, 'dist/sim'));

        const format = Dex.formats.get(formatId);
        if (!format || !format.exists) {
            throw new Error(`Unknown format: ${formatId}`);
        }

        const dex = Dex.forFormat(format);
        const ruleTable = dex.formats.getRuleTable(format);

        function listTsFiles(dir) {
            const out = [];
            const stack = [dir];
            while (stack.length) {
                const current = stack.pop();
                for (const entry of fs.readdirSync(current, {withFileTypes: true})) {
                    const full = path.join(current, entry.name);
                    if (entry.isDirectory()) {
                        stack.push(full);
                    } else if (entry.isFile() && full.endsWith('.ts')) {
                        out.push(full);
                    }
                }
            }
            return out;
        }

        const itemIds = new Set(
            dex.items
                .all()
                .map(item => item.id)
                .filter(Boolean)
        );

        const referencedItemIds = new Set();
        const stringTokenPattern = /['\"]([a-z0-9]+)['\"]/g;
        for (const sourceRoot of [path.join(repoRoot, 'sim'), path.join(repoRoot, 'data')]) {
            for (const file of listTsFiles(sourceRoot)) {
                const text = fs.readFileSync(file, 'utf8');
                for (const match of text.matchAll(stringTokenPattern)) {
                    const token = match[1];
                    if (itemIds.has(token)) referencedItemIds.add(token);
                }
            }
        }

        function nonstandardAllowed(nonstandardTag) {
            if (!nonstandardTag) return true;
            const tagReason = ruleTable.check('pokemontag:' + toID(String(nonstandardTag)));
            if (tagReason) return false;
            if (tagReason === '') return true;
            const nonexistentReason = ruleTable.check('nonexistent');
            if (nonexistentReason) return false;
            return true;
        }

        function legalSpecies(species) {
            if (!species.exists || !species.id) return false;
            if (species.isNonstandard && !nonstandardAllowed(species.isNonstandard)) return false;
            if (ruleTable.isBannedSpecies(species)) return false;
            if (species.battleOnly && species.baseSpecies !== species.name) return false;
            return true;
        }

        function legalMove(move) {
            if (!move.exists || !move.id) return false;
            const moveReason = ruleTable.check('move:' + move.id);
            if (moveReason) return false;
            const allMovesReason = ruleTable.check('pokemontag:allmoves');
            if (allMovesReason) return false;
            if (move.isNonstandard && !nonstandardAllowed(move.isNonstandard)) return false;
            return true;
        }

        function legalAbility(ability) {
            if (!ability.exists || !ability.id) return false;
            const abilityReason = ruleTable.check('ability:' + ability.id);
            if (abilityReason) return false;
            const allAbilitiesReason = ruleTable.check('pokemontag:allabilities');
            if (allAbilitiesReason) return false;
            if (ability.isNonstandard && !nonstandardAllowed(ability.isNonstandard)) return false;
            return true;
        }

        function legalItem(item) {
            if (!item.exists || !item.id) return false;
            const itemReason = ruleTable.check('item:' + item.id);
            if (itemReason) return false;
            const allItemsReason = ruleTable.check('pokemontag:allitems');
            if (allItemsReason) return false;
            if (item.isNonstandard && !nonstandardAllowed(item.isNonstandard)) return false;
            return true;
        }

        function battleRelevantItem(item) {
            if (item.isPokeball) return false;

            const hasBattleHook = Object.keys(item).some(
                key => /^on[A-Z]/.test(key) && typeof item[key] !== 'undefined'
            );
            const hasShapeOrModeEffect = !!(
                item.megaStone ||
                item.onDrive ||
                item.onMemory ||
                item.zMove ||
                item.onPlate ||
                item.forcedForme ||
                item.isPrimalOrb
            );
            const isRuntimeReferenced = referencedItemIds.has(item.id);

            return hasBattleHook || hasShapeOrModeEffect || isRuntimeReferenced;
        }

        const species = dex.species
            .all()
            .filter(legalSpecies)
            .map(species => species.id)
            .sort();

        const abilities = dex.abilities
            .all()
            .filter(legalAbility)
            .map(ability => ability.id)
            .sort();

        const moves = dex.moves
            .all()
            .filter(legalMove)
            .map(move => move.id)
            .sort();

        const legalItems = dex.items
            .all()
            .filter(legalItem);

        const keptItems = legalItems
            .filter(battleRelevantItem)
            .map(item => item.id)
            .sort();

        const removedItems = legalItems
            .filter(item => !battleRelevantItem(item))
            .map(item => item.id)
            .sort();

        const output = {
            format: format.name,
            formatId,
            mod: format.mod,
            ruleset: format.ruleset,
            banlist: format.banlist,
            counts: {
                species: species.length,
                abilities: abilities.length,
                moves: moves.length,
                items: keptItems.length,
                removed_nonbattle_items: removedItems.length,
            },
            entity_tokens: {
                species,
                abilities,
                moves,
                items: keptItems,
            },
            removed_nonbattle_items: removedItems,
            item_filter: 'format-legal AND battle-relevant (event hooks/form effects/runtime references), excluding pokeballs',
        };

        process.stdout.write(JSON.stringify(output));
        """
    )

    completed = subprocess.run(
        ["node", "-e", js_code, str(repo_root), format_id],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return json.loads(completed.stdout)


def _write_lines(path: Path, entries: Sequence[str]) -> None:
    """_write_lines utility used by the RL self-play workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def generate(format_id: str) -> None:
    """generate utility used by the RL self-play workflow."""
    payload = _run_node_vocab_builder(REPO_ROOT, format_id)

    tokens = payload["entity_tokens"]
    counts = payload["counts"]

    _write_lines(RL_DATA_DIR / "sv_ou_species.txt", tokens["species"])
    _write_lines(RL_DATA_DIR / "sv_ou_abilities.txt", tokens["abilities"])
    _write_lines(RL_DATA_DIR / "sv_ou_moves.txt", tokens["moves"])
    _write_lines(RL_DATA_DIR / "sv_ou_items.txt", tokens["items"])
    _write_lines(RL_DATA_DIR / "sv_ou_removed_nonbattle_items.txt", payload["removed_nonbattle_items"])

    source_payload = {
        "format": payload["format"],
        "formatId": payload["formatId"],
        "mod": payload["mod"],
        "ruleset": payload["ruleset"],
        "banlist": payload["banlist"],
        "item_filter": payload["item_filter"],
    }
    (RL_DATA_DIR / "sv_ou_vocab_source.json").write_text(
        json.dumps(source_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (RL_DATA_DIR / "sv_ou_vocab_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """main utility used by the RL self-play workflow."""
    parser = argparse.ArgumentParser(description="Generate Showdown-backed RL entity vocabularies.")
    parser.add_argument("--format", default="gen9ou", help="Showdown format id, for example gen9ou")
    args = parser.parse_args()
    generate(args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
