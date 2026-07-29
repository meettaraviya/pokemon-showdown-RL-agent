# Python Self-Play Bridge (OU)

This directory contains a Python interface to a Pokemon Showdown simulator bridge for `gen9ou`.
It now supports both:

- full self-play rollouts (`run_episode`, `run_batch`)
- stepwise battle control with explicit legal actions for each player
- replay controls for RL (`undo`, `redo`, `restore`, `fork`, and simulation without commit)

Team inputs may be provided as:

- packed team strings
- standard Pokemon Showdown export text
- paths to files containing standard export text

## Files

- `ou_selfplay_client.py`: Python client wrapper and demo entry point.
- `sim/examples/ou-selfplay-bridge.ts`: TypeScript bridge process (JSON lines over stdin/stdout).

## Build

From repository root:

```bash
node build
```

This compiles TypeScript into `.sim-dist`, including:

- `.sim-dist/examples/ou-selfplay-bridge`

## Run demo

From repository root:

```bash
python3 tools/python/ou_selfplay_client.py
```

The demo creates one battle, repeatedly picks the first legal action for both
players, and prints the final result.

You can also run explicit demo mode:

```bash
python3 tools/python/ou_selfplay_client.py demo --num-battles 3 --verbose
```

Add `--simple-state` to collapse team-preview permutations into lead-only choices.

## Run interactive debug CLI

```bash
python3 tools/python/ou_selfplay_client.py debug
```

Use `--simple-state` in debug mode when you want the RL-facing team-preview
interface instead of the raw permutation space.

The debug mode exposes the same control surface intended for RL code:

- inspect RL-facing state (`rl`, `state`, `actions`)
- commit decisions (`choose`, `choosei`, `stepauto`)
- rewind/replay (`undo`, `redo`, `goto`, `history`)
- branchless simulation (`simulate`, `simulatei`, `simstepauto`)
- inspect player-visible information (`state p1`, `requests p1`, `pokemon p1`)

## Python usage

```python
from pathlib import Path
from tools.python.ou_selfplay_client import OUSelfPlayClient

repo_root = Path("/path/to/pokemon-showdown")
sample_team = repo_root / "tools/python/sample_teams/sv-ou/dragmag_hazard_ho.txt"
client = OUSelfPlayClient(repo_root=repo_root)

try:
    battle = client.create_battle(
        include_log=False,
        p1team=sample_team,
        p2team=sample_team.read_text(encoding="utf-8"),
    )

    while not battle.state.done:
        actions = {}
        for player in battle.state.pending_players:
            decision = battle.state.decision_for(player)
            assert decision is not None
            actions[player] = decision.actions[0]
        battle.step(actions)

    print(battle.state.winner, battle.state.turns)

    # Time-travel controls for RL search / planning
    first_pending = battle.state.pending_players[0]
    action = battle.possible_actions(first_pending)[0]

    # Predict next state without mutating live environment
    simulated = battle.simulate_choose(first_pending, action)
    print("simulated turn", simulated.turns, "live turn", battle.state.turns)

    # Commit and then rewind
    battle.choose(first_pending, action)
    battle.undo()

    # Restore to an exact action index in history
    battle.restore(0)

    # Fork a branch battle at current cursor
    branch = battle.fork()
    try:
        # explore alternate line on branch without touching main battle
        if branch.state.pending_players:
            alt_player = branch.state.pending_players[0]
            alt_actions = branch.possible_actions(alt_player)
            if alt_actions:
                branch.choose(alt_player, alt_actions[-1])
    finally:
        branch.close()

    battle.close()
finally:
    client.close()
```

Each pending decision exposes:

- the raw Showdown `request` object
- `actions`, a list of legal simulator commands such as `move 1`, `switch 3`,
  or `team 213456`

This is intended to make it easy to plug a Python policy or reinforcement
learning loop directly into the simulator.

If you want to load a team file yourself, the client also exposes:

```python
team_text = client.read_team(repo_root / "tools/python/sample_teams/sv-ou/balanced_terrain.txt")
```

The older rollout helpers are still available:

```python
episode = client.run_episode(
    include_log=False,
    p1team=repo_root / "tools/python/sample_teams/sv-ou/av_tornadus_specs_kyurem.txt",
)
batch = client.run_batch(100)
```

## Offline and batch RL workflow

The isolated RL package now supports three workflows:

1. Collect an offline dataset from self-play trajectories.
2. Train a permutation-aware policy/value model on that dataset.
3. Evaluate a saved checkpoint with optional planning via simulation branches.

Collect dataset (works without PyTorch):

```bash
python3 tools/python/rl/train.py collect \
    --episodes 500 \
    --output tools/python/rl/data/offline_gen9ou.npz \
    --strategy random \
    --shaping-scale 0.4
```

Train offline model (requires PyTorch):

```bash
python3 tools/python/rl/train.py train \
    --dataset tools/python/rl/data/offline_gen9ou.npz \
    --output tools/python/rl/checkpoints/policy_value.pt \
    --architecture deepset \
    --epochs 30 \
    --batch-size 256
```

Evaluate with planning (requires PyTorch):

```bash
python3 tools/python/rl/train.py eval \
    --checkpoint tools/python/rl/checkpoints/policy_value.pt \
    --episodes 50 \
    --strategy planned \
    --planning-top-k 4
```

Trainer modes and options:

- `collect`
    - `--episodes`: number of battles to sample into offline data.
    - `--output`: destination `.npz` file.
    - `--strategy`: behavior policy for data generation (`random`, `first`).
    - `--shaping-scale`: dense reward shaping coefficient.
    - `--max-actions`: cap on legal action tensor width.
    - `--full-state`: disable simplified team-preview abstraction.
    - `--p1team`, `--p2team`: optional team files for controlled self-play.
- `train`
    - `--dataset`: offline `.npz` generated by collect mode.
    - `--output`: checkpoint destination.
    - `--architecture`: `deepset` (permutation invariant) or `transformer`.
    - `--epochs`, `--batch-size`, `--lr`, `--gamma`.
    - `--value-coef`, `--entropy-coef`: actor/critic and exploration tradeoffs.
    - `--device`: PyTorch device string (for example `cpu`, `cuda`).
- `eval`
    - `--checkpoint`: saved model checkpoint.
    - `--strategy`: `greedy` model action or `planned` one-step planning.
    - `--planning-top-k`: number of top policy actions to simulate.
    - `--planning-gamma`: lookahead discount.
    - `--episodes`, `--max-actions`, `--full-state`, team args.

## Bridge protocol (JSONL)

Each request is one JSON object per line on stdin. Each response is one JSON object per line on stdout.

Commands:

- `{"type":"ping"}`
- `{"type":"create_battle","formatid":"gen9ou","includeLog":false}`
- `{"type":"get_battle_state","battleId":"battle-1"}`
- `{"type":"choose_action","battleId":"battle-1","player":"p1","action":"move 1"}`
- `{"type":"close_battle","battleId":"battle-1"}`
- `{"type":"run_episode","formatid":"gen9ou","includeLog":false}`
- `{"type":"run_batch","count":10,"formatid":"gen9ou","includeLog":false}`

The Python client layers deterministic replay over these commands by creating
new bridge battles with a fixed seed and replaying recorded actions to reach a
target cursor.

The stepwise battle commands and `run_episode` optionally accept `p1team`,
`p2team`, and `seed` (a four-int array). `run_batch` accepts `p1team` and
`p2team`.