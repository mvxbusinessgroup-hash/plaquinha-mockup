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


def brand_colors(img):
    """Retorna (cor_principal, cor_secundaria|None) da logo."""
    a = np.array(img.convert("RGBA"))
    px = a[a[..., 3] > 200][:, :3]
    if len(px) < 50:
        return (60, 60, 60), None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sel = px[(hsv[:, 1] > 45) & (hsv[:, 2] > 60)]
    if len(sel) < max(30, 0.03 * len(px)):
        return (60, 60, 60), None
    sel = sel[:: max(1, len(sel) // 4000)].astype(np.float32)
    k = min(4, len(sel))
    _, labels, centers = cv2.kmeans(sel, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(labels.ravel(), minlength=k).astype(np.float32)
    sat = cv2.cvtColor(np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)[:, 1].astype(np.float32)
    order = np.argsort(-(cnt * (sat / 255) ** 2))
    cols = [tuple(int(c) for c in centers[i]) for i in order]
    main = cols[0]
    sec = None
    for c in cols[1:]:
        if np.abs(np.array(c) - np.array(main)).sum() > 160:
            sec = c
            break
    return main, sec


def brand_color(img):
    return brand_colors(img)[0]


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def shade(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def mix(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


# ---------------------------------------------------------------- arte
def _glyph(d, i, x, y, s, col):
    lw = 4 * s
    if i == 0:   # estrela cheia
        pts = []
        for k in range(10):
            r = (26 if k % 2 == 0 else 11) * s
            ang = np.pi / 2 + k * np.pi / 5
            pts.append((x + r * np.cos(ang), y - r * np.sin(ang) + 2 * s))
        d.polygon(pts, fill=col)
    elif i == 1:  # wi-fi
        for r in (10, 20, 30):
            d.arc([x - r * s, y + 12 * s - r * s, x + r * s, y + 12 * s + r * s], 225, 315, fill=col, width=lw)
        d.ellipse([x - 4 * s, y + 8 * s, x + 4 * s, y + 16 * s], fill=col)
    elif i == 2:  # instagram
        d.rounded_rectangle([x - 22 * s, y - 22 * s, x + 22 * s, y + 22 * s], 12 * s, outline=col, width=lw)
        d.ellipse([x - 10 * s, y - 10 * s, x + 10 * s, y + 10 * s], outline=col, width=lw)
        d.ellipse([x + 11 * s, y - 16 * s, x + 17 * s, y - 10 * s], fill=col)
    elif i == 3:  # whatsapp
        d.ellipse([x - 22 * s, y - 22 * s, x + 22 * s, y + 20 * s], outline=col, width=lw)
        d.polygon([(x - 18 * s, y + 8 * s), (x - 24 * s, y + 26 * s), (x - 6 * s, y + 18 * s)], fill=col)
        d.rounded_rectangle([x - 9 * s, y - 10 * s, x - 3 * s, y + 2 * s], 3 * s, fill=col)
        d.line([(x - 6 * s, y + 0 * s), (x + 6 * s, y + 9 * s)], fill=col, width=lw)
        d.rounded_rectangle([x + 3 * s, y + 4 * s, x + 11 * s, y + 10 * s], 3 * s, fill=col)
    else:          # pix
        for dx, dy in ((0, -13), (13, 0), (0, 13), (-13, 0)):
            cxp, cyp = x + dx * s, y + dy * s
            d.polygon([(cxp, cyp - 9 * s), (cxp + 9 * s, cyp), (cxp, cyp + 9 * s), (cxp - 9 * s, cyp)], fill=col)


def _shadow(im, box, radius, s, strength=70, blur=18, dy=10):
    sh = Image.new("L", im.size, 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(sh).rounded_rectangle([x0, y0 + dy * s, x1, y1 + dy * s], radius, fill=strength)
    from PIL import ImageFilter
    sh = sh.filter(ImageFilter.GaussianBlur(blur * s))
    im.paste((0, 0, 0), (0, 0), sh)


def _fit(img, w, h, cover=False):
    ar, br = img.width / img.height, w / h
    if (ar > br) == cover:
        nh = h; nw = int(h * ar)
    else:
        nw = w; nh = int(w / ar)
    img = img.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
    if cover:
        l, t = (img.width - w) // 2, (img.height - h) // 2
        img = img.crop((l, t, l + w, t + h))
    return img


def make_design(logo_src, nome=None):
    raw = load_image(logo_src)
    logo, complex_bg = clean_logo(raw)
    main, sec = brand_colors(logo if not complex_bg else raw)
    if lum(main) > 225 and (max(main) - min(main)) < 40:   # cor quase branca: usa a secundária
        main, sec = (sec or (40, 40, 40)), None
    s = SS
    Wp, Hp = W * s, H * s
    light = lum(main) > 160
    txt = (25, 30, 45) if light else (255, 255, 255)
    accent = sec if (sec and abs(lum(sec) - lum((255, 255, 255))) > 60) else (shade(main, 0.55) if light else main)
    if lum(accent) > 200:
        accent = (35, 40, 55)

    # fundo em degradê da cor da marca
    top, bot = shade(main, 1.08), shade(main, 0.78)
    g = np.linspace(0, 1, Hp)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - g) + np.array(bot)[None, None, :] * g).repeat(Wp, axis=1)
    im = Image.fromarray(arr.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    # formas decorativas suaves
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.ellipse([(760) * s, (-180) * s, (1260) * s, (320) * s], fill=(255, 255, 255, 38))
    od.ellipse([(-260) * s, (1180) * s, (240) * s, (1680) * s], fill=(255, 255, 255, 30))
    od.ellipse([(-140) * s, (380) * s, (60) * s, (580) * s], fill=(0, 0, 0, 22))
    im.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(im)

    # cartão da logo
    cw, ch = 820, 400
    x0, y0 = (W - cw) // 2, 60
    box = [x0 * s, y0 * s, (x0 + cw) * s, (y0 + ch) * s]
    _shadow(im, box, 44 * s, s)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, 44 * s, fill="white")
    if complex_bg:
        ph = _fit(raw.convert("RGB"), (cw - 24) * s, (ch - 24) * s, cover=True)
        m = Image.new("L", ph.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 34 * s, fill=255)
        im.paste(ph, ((x0 + 12) * s, (y0 + 12) * s), m)
    else:
        bgcard = (35, 35, 40) if lum(np.array(logo)[np.array(logo)[..., 3] > 200][:, :3].mean(0)) > 215 else None
        if bgcard:
            d.rounded_rectangle(box, 44 * s, fill=bgcard)
        lg = _fit(logo, (cw - 120) * s, (ch - 90) * s)
        im.paste(lg, ((W * s - lg.width) // 2, (y0 + ch // 2) * s - lg.height // 2), lg)
    d = ImageDraw.Draw(im)

    # nome
    y = y0 + ch + 40
    if nome:
        f = font("Bold", 58)
        t = nome.upper()[:30]
        while d.textlength(t, font=f) > 880 * s and f.size > 26 * s:
            f = ImageFont.truetype(A("Poppins-Bold.ttf"), f.size - 2 * s)
        d.text((Wp / 2 - d.textlength(t, font=f) / 2, y * s), t, font=f, fill=txt)
        y += 90
    else:
        y += 20

    # cartão branco de baixo
    by0, by1 = y + 10, H - 50
    box2 = [60 * s, by0 * s, (W - 60) * s, by1 * s]
    _shadow(im, box2, 48 * s, s, strength=60)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box2, 48 * s, fill="white")

    # faixa NFC
    ny = by0 + 70
    nr = 46
    d.ellipse([(170 - nr) * s, (ny - nr) * s, (170 + nr) * s, (ny + nr) * s], fill=accent)
    wc = (255, 255, 255)
    d.rounded_rectangle([(176) * s, (ny - 28) * s, (200) * s, (ny + 28) * s], 6 * s, outline=wc, width=4 * s)
    for rr in (10, 20, 30):
        d.arc([(166 - rr) * s, (ny - rr) * s, (166 + rr) * s, (ny + rr) * s], 140, 220, fill=wc, width=4 * s)
    f1 = font("Bold", 40)
    f2 = font("Regular", 26)
    d.text((240 * s, (ny - 44) * s), "Aproxime o celular", font=f1, fill=(30, 32, 40))
    d.text((240 * s, (ny + 6) * s), "ou escaneie o QR Code abaixo", font=f2, fill=(110, 112, 120))

    # ícones
    iy = ny + 140
    labels = ["Avalie", "Wi-Fi", "Instagram", "WhatsApp", "Pix"]
    fl = font("Medium", 22)
    gap = 160
    for i, lab in enumerate(labels):
        x = 500 + (i - 2) * gap
        r = 44
        d.ellipse([(x - r) * s, (iy - r) * s, (x + r) * s, (iy + r) * s], fill=mix(accent, (255, 255, 255), 0.88))
        _glyph(d, i, x * s, iy * s, s, accent)
        d.text((x * s - d.textlength(lab, font=fl) / 2, (iy + r + 10) * s), lab, font=fl, fill=(60, 62, 70))

    # QR
    avail = by1 - (iy + 105) - 90
    qs = int(min(400, avail))
    qy = iy + 105 + max(0, (avail - qs) // 2)
    qx = 500 - qs // 2
    d.rounded_rectangle([qx * s, qy * s, (qx + qs) * s, (qy + qs) * s], 30 * s, fill="white", outline=accent, width=6 * s)
    qr = Image.open(A("qr.png")).convert("RGB").resize(((qs - 60) * s, (qs - 60) * s), Image.NEAREST)
    im.paste(qr, ((qx + 30) * s, (qy + 30) * s))
    f = font("Bold", 32)
    t = "Escaneie o QR Code"
    d.text((Wp / 2 - d.textlength(t, font=f) / 2, (qy + qs + 18) * s), t, font=f, fill=(30, 32, 40))

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
