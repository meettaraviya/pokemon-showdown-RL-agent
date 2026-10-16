# AI Battle Client README

This document describes how to host the RL web dashboard you added on top of Pokemon Showdown.

It is based on:
- Your RL commit: `0080e7409` (Implemented RL with learning from self-play)
- Your current local UI server and frontend files:
  - `tools/python/rl/ui_server.py`
  - `tools/python/rl/ui/index.html`
  - `tools/python/rl/ui/app.js`

## What This Website Is

The website is a standalone local web app served directly by Python:
- Static frontend: `tools/python/rl/ui/*`
- JSON API: `/api/*` routes from `tools/python/rl/ui_server.py`
- Backend actions:
  - Start PPO training jobs
  - Start PPO evaluation jobs
  - Create and inspect live battles
  - Step through battle history and view model/action analysis

The frontend uses relative `/api/...` requests, so static pages and API must be on the same origin.

## Prerequisites

From repo root:

1. Build Pokemon Showdown TypeScript output (needed by the bridge process):

```bash
node build
```

2. Use Python 3.10+ (3.11+ recommended) and install dependencies in your environment.

Minimum required Python packages for the UI + RL workflows:

```bash
python3 -m pip install numpy torch
```

If you only want non-model battle browsing, torch is still recommended because model checkpoint features and some training/eval flows require it.

## Run Locally (Fastest)

From repo root:

```bash
python3 tools/python/rl/ui_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Useful local development mode (auto-restarts on Python/UI file changes):

```bash
python3 tools/python/rl/ui_server.py --host 127.0.0.1 --port 8765 --reload
```

Optional extra watch paths (repeatable):

```bash
python3 tools/python/rl/ui_server.py --reload --reload-path sim/examples --reload-path tools/python/rl
```

## Host on a Server (LAN/VPS)

### 1) Bind to all interfaces

```bash
python3 tools/python/rl/ui_server.py --host 0.0.0.0 --port 8765
```

Then access from another machine:

```text
http://<server-ip>:8765/
```

Security note: the API can launch training/eval jobs and battles. Do not expose this publicly without access control.

### 2) Keep it running as a service (systemd)

Create `/etc/systemd/system/ps-rl-ui.service`:

```ini
[Unit]
Description=Pokemon Showdown RL UI
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/pokemon-showdown
ExecStart=/usr/bin/python3 /root/pokemon-showdown/tools/python/rl/ui_server.py --host 127.0.0.1 --port 8765
Restart=always
RestartSec=2
User=root
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ps-rl-ui
sudo systemctl status ps-rl-ui
```

Logs:

```bash
sudo journalctl -u ps-rl-ui -f
```

### 3) Put Nginx in front (recommended)

Example site config:

```nginx
server {
    listen 80;
    server_name your-domain.example;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then add TLS (for public internet) with your usual Certbot workflow.

## Common Startup Workflow

From a fresh pull:

```bash
cd /root/pokemon-showdown
node build
python3 tools/python/rl/ui_server.py --host 127.0.0.1 --port 8765
```

## Troubleshooting

- `ModuleNotFoundError` or torch-related import errors:
  - Install dependencies into the same Python environment used to start `ui_server.py`.

- UI loads but actions fail:
  - Ensure `node build` completed so bridge artifacts are available.

- External machine cannot connect:
  - Start with `--host 0.0.0.0` and open port `8765` in firewall/security group.

- Checkpoints/teams not listed:
  - The UI enumerates paths under:
    - `tools/python/sample_teams`
    - `tools/python/rl/checkpoints`

## API Routes (for health checks/integration)

Read endpoints:
- `GET /api/meta`
- `GET /api/jobs`
- `GET /api/jobs/{jobId}`
- `GET /api/battles`
- `GET /api/battles/{battleId}?cursor=<n>&player=p1|p2`

Write endpoints:
- `POST /api/train/ppo`
- `POST /api/eval/ppo`
- `POST /api/battles`
- `POST /api/battles/{battleId}/advance`
- `POST /api/battles/{battleId}/choose`
- `POST /api/battles/{battleId}/close`

---

If you later split frontend and backend origins, you will need to add CORS handling in `ui_server.py`. Right now the app expects same-origin hosting.
