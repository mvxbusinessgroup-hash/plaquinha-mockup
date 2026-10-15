"""Servidor que gera o mockup 3D da plaquinha NFC automaticamente.

Rotas:
  POST /zapdata        -> recebe o webhook do ZapData, acha a logo do lead e devolve {"ok", "url", ...}
  GET|POST /mockup     -> gera a partir de logo_url (+ nome) — uso manual
  GET  /               -> página simples para gerar subindo a logo
  GET  /img/<arquivo>  -> imagens geradas
  GET  /debug/last     -> último payload recebido do ZapData (precisa de ?key=)
"""
import json
import threading
import os
import re
import time
import urllib.request
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import generator

OUT = os.environ.get("OUT_DIR", "/tmp/out")
os.makedirs(OUT, exist_ok=True)
API_KEY = os.environ.get("API_KEY", "")          # opcional: exige ?key= no webhook
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
SHOW_NAME = os.environ.get("SHOW_NAME", "0") == "1"
LAST = {"payload": None, "at": None, "result": None}

app = FastAPI(title="Mockup Plaquinha NFC")

IMG_EXT = re.compile(r"https?://[^\s\"'<>]+", re.I)


def base_url(request: Request) -> str:
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host"))
    return f"{proto}://{host}"


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path.lower(), obj


def is_image(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "PlaquinhaMockup/1.0 (mockup generator)"})
        with urllib.request.urlopen(req, timeout=6) as r:
            ct = r.headers.get("Content-Type", "")
            head = r.read(16)
        return ct.startswith("image/") or head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or head[:4] == b"RIFF"
    except Exception:
        return False


def find_logo(payload):
    """Usa só a imagem (logo) que o lead mandou."""
    items = list(walk(payload))
    urls, pics, texts = [], [], []
    for path, v in items:
        if not isinstance(v, str):
            continue
        texts.append(v)
        for u in IMG_EXT.findall(v):
            if any(k in path for k in ("pic", "avatar", "photo", "profile")):
                pics.append(u)
            elif "instagram.com" not in u:
                urls.append(u)
    for u in urls:                       # mídia enviada (logo)
        if is_image(u):
            return u, "imagem"
    return None, None


def find_name(payload):
    for path, v in walk(payload):
        if isinstance(v, str) and v.strip() and path.endswith(("contactname", ".nome", ".name", "pushname")):
            return v.strip()
    return None


def _do_render(src, nome, fid):
    try:
        mock, arte = generator.generate(src, nome if SHOW_NAME else None)
        open(os.path.join(OUT, f"{fid}_arte.png"), "wb").write(arte)
        tmp = os.path.join(OUT, f"{fid}.tmp")
        open(tmp, "wb").write(mock)
        os.replace(tmp, os.path.join(OUT, f"{fid}.jpg"))
    except Exception as e:
        open(os.path.join(OUT, f"{fid}.err"), "w").write(str(e)[:300])


def render(src, nome, request, background=False):
    fid = uuid.uuid4().hex[:12]
    b = base_url(request)
    res = {"ok": True, "url": f"{b}/img/{fid}.jpg", "arte_url": f"{b}/img/{fid}_arte.png"}
    if background:   # responde na hora (o ZapData tem timeout curto) e gera em paralelo
        threading.Thread(target=_do_render, args=(src, nome, fid), daemon=True).start()
    else:
        _do_render(src, nome, fid)
        if os.path.exists(os.path.join(OUT, f"{fid}.err")):
            raise RuntimeError(open(os.path.join(OUT, f"{fid}.err")).read())
    return res


@app.post("/zapdata")
async def zapdata(request: Request):
    if API_KEY and request.query_params.get("key") != API_KEY:
        return JSONResponse({"ok": False, "erro": "chave inválida"}, status_code=401)
    try:
        payload = await request.json()
    except Exception:
        payload = {"raw": (await request.body()).decode("utf-8", "ignore")}
    LAST.update(payload=payload, at=time.strftime("%Y-%m-%d %H:%M:%S"))
    src, fonte = find_logo(payload)
    if not src:
        res = {"ok": False, "fonte": "nenhuma", "url": ""}
    else:
        try:
            res = render(src, find_name(payload), request, background=True)
            res["fonte"] = fonte
        except Exception as e:  # logo inválida
            res = {"ok": False, "fonte": fonte, "erro": str(e)[:200], "url": ""}
    LAST["result"] = res
    return res


@app.api_route("/mockup", methods=["GET", "POST"])
async def mockup(request: Request):
    params = dict(request.query_params)
    if request.method == "POST":
        try:
            params.update(await request.json())
        except Exception:
            pass
    src = params.get("logo_url")
    if not src:
        return JSONResponse({"ok": False, "erro": "envie logo_url"}, status_code=400)
    return render(src, params.get("nome"), request)


@app.post("/upload")
async def upload(request: Request, logo: UploadFile = File(...), nome: str = Form("")):
    res = render(await logo.read(), nome or None, request)
    return HTMLResponse(f'<p><a href="/">voltar</a></p><img src="{res["url"]}" style="max-width:100%"><p><a href="{res["url"]}" download>Baixar mockup</a> · <a href="{res["arte_url"]}" download>Baixar arte</a></p>')


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Mockup Plaquinha NFC</title>
<body style="font-family:system-ui;max-width:520px;margin:40px auto;padding:0 16px">
<h2>Gerar mockup da plaquinha</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
<p>Logo: <input type="file" name="logo" accept="image/*" required></p>
<p>Nome da loja: <input name="nome" style="width:100%"></p>
<p><button>Gerar</button></p></form></body>"""


@app.get("/img/{name}")
def img(name: str):
    path = os.path.join(OUT, os.path.basename(name))
    fid = os.path.basename(name).split(".")[0].replace("_arte", "")
    for _ in range(90):   # espera a geração em segundo plano (até ~45s)
        if os.path.exists(path) or os.path.exists(os.path.join(OUT, f"{fid}.err")):
            break
        time.sleep(0.5)
    if not os.path.exists(path):
        return JSONResponse({"erro": "não encontrado"}, status_code=404)
    return FileResponse(path)


@app.get("/debug/last")
def debug_last(key: str = ""):
    if API_KEY and key != API_KEY:
        return JSONResponse({"erro": "chave inválida"}, status_code=401)
    return JSONResponse(json.loads(json.dumps(LAST, default=str)))
