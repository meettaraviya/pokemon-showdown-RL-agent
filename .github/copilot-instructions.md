Pokémon Showdown
========================================================================

Navigation: [Website][1] | **Server repository** | [Client repository][2] | [Dex repository][3]

  [1]: http://pokemonshowdown.com/
  [2]: https://github.com/smogon/pokemon-showdown-client
  [3]: https://github.com/Zarel/Pokemon-Showdown-Dex

[![Build Status](https://github.com/smogon/pokemon-showdown/workflows/Node.js%20CI/badge.svg)](https://github.com/smogon/pokemon-showdown/actions?query=workflow%3A%22Node.js+CI%22)
[![Dependency Status](https://img.shields.io/librariesio/github/smogon/pokemon-showdown)](https://libraries.io/github/smogon/pokemon-showdown)


Introduction
------------------------------------------------------------------------

Pokémon Showdown is many things:

- A **web site** you can use for Pokémon battling

  - http://pokemonshowdown.com/

- A **JavaScript library** for simulating Pokémon battles and getting Pokédex data

  - [sim/README.md](./sim/README.md)

- Some **command-line tools** for simulating Pokémon battles (which can be used in non-JavaScript programs)

  - [COMMANDLINE.md](./COMMANDLINE.md)

- A **web API** for the web site for Pokémon battling

  - [pokemon-showdown-client: WEB-API.md](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md)

- A **game server** for hosting your own Pokémon Showdown community and game modes

  - [server/README.md](./server/README.md)

Pokémon Showdown simulates singles, doubles and triples battles in all the games out so far (Generations 1 through 9).


Repository-specific focus: RL battle agent development
------------------------------------------------------------------------

This fork's primary goal is reinforcement learning for Pokemon battles. Treat RL
agent quality, training velocity, and simulator control surface improvements as
the highest-priority outcomes for code changes.

Key RL stack and workflows:

- Simulator bridge for Python RL integration:
  - `sim/examples/ou-selfplay-bridge.ts`
  - JSONL bridge over stdin/stdout for creating battles, stepping actions, and
    collecting state/results.
- Python bridge client and deterministic battle controls:
  - `tools/python/ou_selfplay_client.py`
  - Supports legal action enumeration and time-travel/replay controls used by
    planning and training (`undo`, `redo`, `restore`, `fork`, simulation
    without committing live state).
- RL package:
  - `tools/python/rl/`
  - Offline data collection/training/evaluation and PPO training/evaluation are
    driven from `tools/python/rl/train.py`.
  - Environment/features/model/planning code lives in
    `environment.py`, `features.py`, `action_features.py`, `models.py`,
    `planning.py`, `offline_dataset.py`, and `replay.py`.
- Local RL web dashboard:
  - `tools/python/rl/ui_server.py` with static UI under `tools/python/rl/ui/`
  - Used to launch training/eval jobs, inspect live battles, and review
    model/action analysis.
- Team and artifact paths used by RL workflows:
  - `tools/python/sample_teams/`
  - `tools/python/rl/checkpoints/`
  - `tools/python/rl/data/`

Working expectations for this repository:

- Prefer changes that improve agent learning signal quality, action selection,
  planning, reproducibility, and debugging visibility.
- Preserve and extend RL-facing simulator capabilities before optimizing
  unrelated server features.
- Keep bridge/client behavior deterministic where possible to support stable
  experiments and reproducible evaluations.
- When changing simulator interfaces used by the RL stack, update Python
  integration points and related docs together.


Documentation quick links
------------------------------------------------------------------------

* [PROTOCOL.md][4] - How the client and server communicate with each other.
* [sim/SIM-PROTOCOL.md][5] - The part of the protocol used for battles and battle messages.
* [CONTRIBUTING.md][6] - Useful code standards to understand if you want to send pull requests to PS (not necessary if you're just using the code and not planning to contribute back).
* [ARCHITECTURE.md][7] - A high-level overview of how the code works.
* [Bot FAQ][8] - An FAQ compiled by Kaiepi regarding making Pokemon Showdown bots - mainly chatbots and battle bots.

  [4]: ./PROTOCOL.md
  [5]: ./sim/SIM-PROTOCOL.md
  [6]: ./CONTRIBUTING.md
  [7]: ./ARCHITECTURE.md
  [8]: https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e

<!-- graphify-rules-start (managed by `graphify init`) -->
Use Graphify before grep
------------------------------------------------------------------------

This repository is indexed by Graphify: a code graph over its call, dependency, and test structure, exposed through a connected Graphify MCP server. Before reaching for grep or reading files, use the Graphify tools your MCP client lists (their exact names and descriptions are in the server's tool list) for what the graph knows and a text search does not:

- find where a symbol, function, or class is defined (instead of grepping for it)
- understand how something works, or where a behavior is handled
- find who calls a function, or what it calls
- see what a change affects (its blast radius) and which tests cover it
- map a file's dependencies and dependents

Fall back to grep or file reads only for what the graph does not model: literal string or comment matches, non-indexed files, or reading a file you have already located. If no Graphify tools are listed, check the MCP server connection.
<!-- graphify-rules-end -->