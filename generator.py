"""Gera a arte da plaquinha NFC a partir de uma logo e o mockup 3D no display de acrílico."""
import io
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda f: os.path.join(HERE, f)

W, H = 1000, 1500          # arte 10x15 cm
SS = 2                     # supersampling para bordas suaves
INK = (45, 40, 38)
QUAD = np.float32([[153, 61], [728, 59], [836, 904], [292, 1037]])  # frente do acrílico em base.jpg


def font(name, size):
    return ImageFont.truetype(A(f"Poppins-{name}.ttf"), size * SS)


# ---------------------------------------------------------------- logo
def load_image(src):
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    elif str(src).startswith("http"):
        req = urllib.request.Request(src, headers={"User-Agent": "PlaquinhaMockup/1.0 (mockup generator)"})
        data = urllib.request.urlopen(req, timeout=20).read()
    else:
        data = open(src, "rb").read()
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    im.thumbnail((1000, 1000), Image.LANCZOS)
    return im


def clean_logo(img):
    """Remove fundo liso (branco ou cor única) e recorta. Retorna (RGBA, tem_fundo_complexo)."""
    a = np.array(img)
    alpha = a[..., 3]
    if (alpha < 250).mean() > 0.02:          # já tem transparência
        out = a
    else:
        rgb = a[..., :3].astype(np.int16)
        h, w = alpha.shape
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bgc = np.median(border, axis=0)
        if np.abs(border - bgc).sum(1).mean() > 45:   # fundo não é liso (foto)
            return img, True
        diff = (np.abs(rgb - bgc).sum(2) < 60).astype(np.uint8)
        ff = diff.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if ff[y, x] == 1:
                cv2.floodFill(ff, mask, (x, y), 2)
        bgmask = (ff == 2)
        out = a.copy()
        out[..., 3] = np.where(bgmask, 0, 255).astype(np.uint8)
        out[..., 3] = cv2.GaussianBlur(out[..., 3], (3, 3), 0)
    ys, xs = np.nonzero(out[..., 3] > 20)
    if len(xs) == 0:
        return img, True
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(out), False


def brand_color(img):
    a = np.array(img.convert("RGBA"))
    px = a[a[..., 3] > 200][:, :3]
    if len(px) < 50:
        return (90, 90, 90)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sel = px[(hsv[:, 1] > 90) & (hsv[:, 2] > 60)]
    if len(sel) < max(30, 0.03 * len(px)):
        return (90, 90, 90)
    sel = sel[:: max(1, len(sel) // 4000)].astype(np.float32)
    k = min(3, len(sel))
    _, labels, centers = cv2.kmeans(sel, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    best = np.bincount(labels.ravel()).argmax()
    return tuple(int(c) for c in centers[best])


# ---------------------------------------------------------------- arte
def _icons(d, cx, cy, gap, ink):
    s = SS
    lw = 4 * s
    labels = ["Avalie", "Wi-Fi", "Instagram", "WhatsApp", "Pix"]
    f = font("Regular", 22)
    for i, lab in enumerate(labels):
        x = (cx + (i - 2) * gap) * s
        y = cy * s
        if i == 0:   # estrela
            pts = []
            for k in range(10):
                r = (34 if k % 2 == 0 else 15) * s
                ang = np.pi / 2 + k * np.pi / 5
                pts.append((x + r * np.cos(ang), y - r * np.sin(ang)))
            d.polygon(pts, outline=ink, width=lw)
        elif i == 1:  # wi-fi
            for r in (14, 26, 38):
                d.arc([x - r * s, y + 14 * s - r * s, x + r * s, y + 14 * s + r * s], 225, 315, fill=ink, width=lw)
            d.ellipse([x - 5 * s, y + 9 * s, x + 5 * s, y + 19 * s], fill=ink)
        elif i == 2:  # câmera
            d.rounded_rectangle([x - 32 * s, y - 22 * s, x + 32 * s, y + 24 * s], 10 * s, outline=ink, width=lw)
            d.rounded_rectangle([x - 14 * s, y - 31 * s, x + 14 * s, y - 22 * s], 4 * s, fill=ink)
            d.ellipse([x - 13 * s, y - 12 * s, x + 13 * s, y + 14 * s], outline=ink, width=lw)
        elif i == 3:  # balão
            d.ellipse([x - 32 * s, y - 28 * s, x + 32 * s, y + 22 * s], outline=ink, width=lw)
            d.polygon([(x - 26 * s, y + 8 * s), (x - 34 * s, y + 32 * s), (x - 10 * s, y + 20 * s)], fill=ink)
        else:          # losango
            d.polygon([(x, y - 30 * s), (x + 30 * s, y), (x, y + 30 * s), (x - 30 * s, y)], outline=ink, width=lw)
        tw = d.textlength(lab, font=f)
        d.text((x - tw / 2, y + 44 * s), lab, font=f, fill=ink)


def make_design(logo_src, nome=None):
    raw = load_image(logo_src)
    logo, complex_bg = clean_logo(raw)
    brand = brand_color(logo if not complex_bg else raw)
    s = SS
    im = Image.new("RGB", (W * s, H * s), "white")
    d = ImageDraw.Draw(im)
    deco = brand
    # círculos decorativos (iguais aos modelos do Canva)
    d.ellipse([(940 - 292) * s, (35 - 292) * s, (940 + 292) * s, (35 + 292) * s], fill=deco)
    d.ellipse([(56 - 222) * s, (1398 - 222) * s, (56 + 222) * s, (1398 + 222) * s], fill=deco)

    # logo no círculo
    cx, cy, r = 500, 235, 172
    lum = np.array(logo.convert("L"))[np.array(logo)[..., 3] > 200].mean() if not complex_bg else 128
    circle_fill = (35, 35, 35) if lum > 215 else (255, 255, 255)
    ar = raw.width / raw.height
    wide = complex_bg and (ar > 1.25 or ar < 0.8)
    if wide:  # foto retangular (ex.: fachada/placa): mostra inteira num quadro arredondado
        maxw, maxh = 760, 340
        bw, bh = (maxw, maxw / ar) if maxw / ar <= maxh else (maxh * ar, maxh)
        bw, bh = int(bw), int(bh)
        ph = raw.convert("RGB").resize((bw * s, bh * s), Image.LANCZOS)
        m = Image.new("L", ph.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 28 * s, fill=255)
        x0, y0 = (cx - bw // 2) * s, (cy - bh // 2) * s
        im.paste(ph, (x0, y0), m)
        d.rounded_rectangle([x0, y0, x0 + ph.width, y0 + ph.height], 28 * s, outline=INK, width=4 * s)
    else:
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=circle_fill, outline=INK, width=4 * s)
    if wide:
        pass
    elif complex_bg:  # foto: recorte circular cobrindo o círculo
        side = min(raw.size)
        sq = raw.crop(((raw.width - side) // 2, (raw.height - side) // 2, (raw.width + side) // 2, (raw.height + side) // 2))
        sq = sq.resize((2 * (r - 4) * s, 2 * (r - 4) * s), Image.LANCZOS)
        m = Image.new("L", sq.size, 0)
        ImageDraw.Draw(m).ellipse([0, 0, sq.width - 1, sq.height - 1], fill=255)
        im.paste(sq.convert("RGB"), ((cx - r + 4) * s, (cy - r + 4) * s), m)
    else:
        box = int(2 * r * 0.70) * s
        lg = logo.copy()
        lg.thumbnail((box, box), Image.LANCZOS)
        if max(lg.size) < box:  # amplia logos pequenas
            k = box / max(lg.size)
            lg = lg.resize((int(lg.width * k), int(lg.height * k)), Image.LANCZOS)
        im.paste(lg, (cx * s - lg.width // 2, cy * s - lg.height // 2), lg)

    y = 470
    if nome:
        f = font("Bold", 44)
        txt = nome.upper()[:28]
        while d.textlength(txt, font=f) > 860 * s and f.size > 24 * s:
            f = ImageFont.truetype(A("Poppins-Bold.ttf"), f.size - 2 * s)
        tw = d.textlength(txt, font=f)
        d.text((W * s / 2 - tw / 2, (y - 30) * s), txt, font=f, fill=INK)

    # NFC
    ncx, ncy, nr = 500, 665, 100
    d.ellipse([(ncx - nr) * s, (ncy - nr) * s, (ncx + nr) * s, (ncy + nr) * s], outline=INK, width=4 * s)
    d.rounded_rectangle([(ncx + 6) * s, (ncy - 62) * s, (ncx + 64) * s, (ncy + 62) * s], 12 * s, outline=INK, width=4 * s)
    d.line([(ncx + 26) * s, (ncy - 50) * s, (ncx + 44) * s, (ncy - 50) * s], fill=INK, width=4 * s)
    for rr in (18, 36, 54):
        d.arc([(ncx - 14 - rr) * s, (ncy - rr) * s, (ncx - 14 + rr) * s, (ncy + rr) * s], 140, 220, fill=INK, width=5 * s)
    f = font("Regular", 40)
    t = "Aproxime o celular"
    d.text((W * s / 2 - d.textlength(t, font=f) / 2, 780 * s), t, font=f, fill=INK)

    _icons(d, 500, 880, 170, INK)

    # QR
    qs, qy = 344, 990
    d.rounded_rectangle([(500 - qs // 2) * s, qy * s, (500 + qs // 2) * s, (qy + qs) * s], 26 * s, fill="white", outline=INK, width=4 * s)
    qr = Image.open(A("qr.png")).convert("RGB").resize(((qs - 64) * s, (qs - 64) * s), Image.NEAREST)
    im.paste(qr, ((500 - qs // 2 + 32) * s, (qy + 32) * s))
    f = font("Bold", 38)
    t = "Escaneie o QR Code"
    d.text((W * s / 2 - d.textlength(t, font=f) / 2, (qy + qs + 26) * s), t, font=f, fill=INK)

    return im.resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------- mockup 3D
def make_mockup(design_pil):
    bg = cv2.imread(A("base.jpg"))
    Hh, Ww = bg.shape[:2]
    des = cv2.cvtColor(np.array(design_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = des.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), QUAD)
    warped = cv2.warpPerspective(des, M, (Ww, Hh), flags=cv2.INTER_AREA)
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (Ww, Hh))
    m = cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32)[..., None] / 255
    bgf = bg.astype(np.float32) / 255
    wf = warped.astype(np.float32) / 255
    mb = (m[..., 0] * 255).astype(np.uint8)
    inner = cv2.erode(mb, np.ones((12, 12), np.uint8))
    edge = cv2.GaussianBlur((mb.astype(np.float32) - inner.astype(np.float32)) / 255, (9, 9), 0)[..., None]
    shade = np.clip(bgf * 1.04, 0, 1) * edge + (1 - edge)
    comp = bgf * (1 - m) + (wf * shade) * m
    t = np.arange(Ww, dtype=np.float32)[None, :] * 0.55 + np.arange(Hh, dtype=np.float32)[:, None]
    band = np.exp(-((t - 700) / 170) ** 2) * 0.22 + np.exp(-((t - 1080) / 45) ** 2) * 0.12
    comp = comp + (1 - comp) * band[..., None] * m
    out = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def generate(logo_src, nome=None):
    """Retorna (jpg_mockup_bytes, png_arte_bytes)."""
    design = make_design(logo_src, nome)
    b = io.BytesIO()
    design.save(b, "PNG")
    return make_mockup(design), b.getvalue()
