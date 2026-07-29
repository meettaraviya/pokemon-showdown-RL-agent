# SV OU sample teams

These are simulator-ready Pokemon Showdown export files intended for the Python self-play client.

Sources:

- `dragmag_hazard_ho.txt`: https://pokepast.es/57780b9afe26d51b
- `balanced_terrain.txt`: https://pokepast.es/43eb8733a7cff75e
- `av_tornadus_specs_kyurem.txt`: https://pokepast.es/54d94a017c0f67ee

Usage example:

```python
from pathlib import Path
from tools.python.ou_selfplay_client import OUSelfPlayClient

repo_root = Path('/root/pokemon-showdown')
client = OUSelfPlayClient(repo_root=repo_root)

try:
    battle = client.create_battle(
        p1team=repo_root / 'tools/python/sample_teams/sv-ou/dragmag_hazard_ho.txt',
        p2team=repo_root / 'tools/python/sample_teams/sv-ou/balanced_terrain.txt',
    )
finally:
    client.close()
```
