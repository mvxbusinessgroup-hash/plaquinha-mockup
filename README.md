---
title: Mockup Plaquinha NFC
emoji: 🪧
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Gera automaticamente o mockup 3D da plaquinha NFC a partir da logo do cliente.

- `POST /zapdata` — URL para colocar no bloco Webhook do ZapData. Responde `{"ok": true, "url": "<link do mockup>"}`.
- `GET /` — página para gerar manualmente subindo a logo.
