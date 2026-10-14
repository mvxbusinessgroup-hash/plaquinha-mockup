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


def _icon_row(d, cy, s, ink, bg):
    lw = 5 * s
    xs = [500 - 2 * 150, 500 - 150, 500, 500 + 150, 500 + 300]
    # G (Google)
    x = xs[0] * s; y = cy * s
    f = ImageFont.truetype(A("Poppins-Bold.ttf"), 78 * s)
    d.text((x, y + 4 * s), "G", font=f, fill=ink, anchor="mm")
    # Wi-Fi (selo)
    x = xs[1] * s
    d.rounded_rectangle([x - 52 * s, y - 26 * s, x + 52 * s, y + 26 * s], 26 * s, outline=ink, width=lw)
    f2 = ImageFont.truetype(A("Poppins-Bold.ttf"), 30 * s)
    d.text((x, y + 2 * s), "WiFi", font=f2, fill=ink, anchor="mm")
    # Pix
    x = xs[2] * s
    for dx, dy in ((0, -17), (17, 0), (0, 17), (-17, 0)):
        cx, cy2 = x + dx * s, y + dy * s
        r = 13 * s
        d.polygon([(cx, cy2 - r), (cx + r, cy2), (cx, cy2 + r), (cx - r, cy2)], fill=ink)
    # Instagram
    x = xs[3] * s
    d.rounded_rectangle([x - 32 * s, y - 32 * s, x + 32 * s, y + 32 * s], 16 * s, outline=ink, width=lw)
    d.ellipse([x - 14 * s, y - 14 * s, x + 14 * s, y + 14 * s], outline=ink, width=lw)
    d.ellipse([x + 15 * s, y - 23 * s, x + 23 * s, y - 15 * s], fill=ink)
    # WhatsApp
    x = xs[4] * s
    d.ellipse([x - 31 * s, y - 31 * s, x + 31 * s, y + 29 * s], outline=ink, width=lw)
    d.polygon([(x - 25 * s, y + 12 * s), (x - 34 * s, y + 36 * s), (x - 9 * s, y + 26 * s)], fill=ink)
    d.polygon([(x - 22 * s, y + 13 * s), (x - 27 * s, y + 27 * s), (x - 12 * s, y + 21 * s)], fill=bg)
    d.rounded_rectangle([x - 14 * s, y - 15 * s, x - 6 * s, y + 0 * s], 4 * s, fill=ink)
    d.line([(x - 10 * s, y - 2 * s), (x + 8 * s, y + 12 * s)], fill=ink, width=7 * s)
    d.rounded_rectangle([x + 4 * s, y + 6 * s, x + 16 * s, y + 14 * s], 4 * s, fill=ink)


def _nfc_art(d, cx, cy, s, ink, bg):
    lw = 6 * s
    # oval do terminal com ondas
    ex0, ey0, ex1, ey1 = cx - 190, cy - 90, cx + 60, cy + 70
    d.ellipse([ex0 * s, ey0 * s, ex1 * s, ey1 * s], outline=ink, width=lw)
    wx, wy = cx - 95, cy - 10
    for rr in (16, 32, 48):
        d.arc([(wx - rr) * s, (wy - rr) * s, (wx + rr) * s, (wy + rr) * s], -50, 50, fill=ink, width=lw)
    # mão (palma) atrás do celular
    px0, py0, px1, py1 = cx + 5, cy - 115, cx + 115, cy + 95
    d.rounded_rectangle([(px0 - 10) * s, (cy + 20) * s, (px1 + 28) * s, (cy + 150) * s], 46 * s, fill=bg, outline=ink, width=lw)
    # celular
    d.rounded_rectangle([px0 * s, py0 * s, px1 * s, py1 * s], 20 * s, fill=bg, outline=ink, width=lw)
    d.line([((px0 + px1) / 2 - 16) * s, (py0 + 14) * s, ((px0 + px1) / 2 + 16) * s, (py0 + 14) * s], fill=ink, width=4 * s)
    # dedos
    for fy in (cy - 5, cy + 28, cy + 61):
        d.rounded_rectangle([(px0 - 36) * s, fy * s, (px0 + 22) * s, (fy + 30) * s], 15 * s, fill=bg, outline=ink, width=lw)
    # polegar
    d.rounded_rectangle([(px1 - 14) * s, (cy - 20) * s, (px1 + 18) * s, (cy + 50) * s], 16 * s, fill=bg, outline=ink, width=lw)


def make_design(logo_src, nome=None):
    raw = load_image(logo_src)
    logo, complex_bg = clean_logo(raw)
    ra0 = np.array(raw.convert("RGB")).astype(np.float32)
    bmed = np.median(np.concatenate([ra0[0], ra0[-1], ra0[:, 0], ra0[:, -1]]), axis=0)
    if lum(bmed) < 70 and 0.8 <= raw.width / raw.height <= 1.25 and (np.array(raw)[..., 3] > 250).mean() > 0.98:
        complex_bg = True   # logo quadrada com fundo escuro (foto de perfil): usa como selo redondo
    main, sec = brand_colors(logo if not complex_bg else raw)
    if lum(main) > 225 and (max(main) - min(main)) < 40:
        main, sec = (sec or (40, 40, 40)), None
    s = SS
    Wp, Hp = W * s, H * s
    # fundo: tom profundo da cor da marca (estilo acrílico premium)
    sat = max(main) - min(main)
    base = mix(main, (18, 18, 20), 0.45 if sat > 60 else 0.75)
    if complex_bg:   # logo com fundo próprio (ex.: foto de perfil): usa o tom da borda se for escuro
        ra = np.array(raw.convert("RGB")).astype(np.float32)
        border = np.concatenate([ra[0], ra[-1], ra[:, 0], ra[:, -1]])
        bc = tuple(int(v) for v in np.median(border, axis=0))
        if lum(bc) < 70:
            base = mix((16, 16, 18), main, 0.10)
    else:
        a0 = np.array(logo)
        if (a0[..., 3] > 200).any() and lum(a0[a0[..., 3] > 200][:, :3].mean(0)) < 90:
            base = mix((16, 16, 18), main, 0.10)
    ink = (255, 255, 255)
    top, bot = shade(base, 1.15), shade(base, 0.8)
    g = np.linspace(0, 1, Hp)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - g) + np.array(bot)[None, None, :] * g).repeat(Wp, axis=1)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    # ondas decorativas finas no canto
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for k in range(14):
        r = 180 + k * 18
        od.arc([(1000 - r) * s, (-r) * s, (1000 + r) * s, (r) * s], 90, 180, fill=(255, 255, 255, 34), width=2 * s)
        r2 = 140 + k * 18
        od.arc([(-r2) * s, (1500 - r2) * s, (r2) * s, (1500 + r2) * s], 270, 360, fill=(255, 255, 255, 22), width=2 * s)
    im = Image.alpha_composite(im, ov).convert("RGB")
    d = ImageDraw.Draw(im)

    # logo no topo
    lcx, lcy = 500, 200
    ar = raw.width / raw.height
    if complex_bg and 0.8 <= ar <= 1.25:   # foto/selo quadrado: círculo
        dia = 280
        sq = _fit(raw.convert("RGB"), dia * s, dia * s, cover=True)
        m = Image.new("L", sq.size, 0)
        ImageDraw.Draw(m).ellipse([0, 0, sq.width - 1, sq.height - 1], fill=255)
        im.paste(sq, ((lcx - dia // 2) * s, (lcy - dia // 2) * s), m)
    elif complex_bg:
        cw, ch = 540, 270
        ph = _fit(raw.convert("RGB"), cw * s, ch * s, cover=True)
        m = Image.new("L", ph.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 30 * s, fill=255)
        im.paste(ph, ((lcx - cw // 2) * s, (lcy - ch // 2) * s), m)
        d.rounded_rectangle([(lcx - cw // 2) * s, (lcy - ch // 2) * s, (lcx + cw // 2) * s, (lcy + ch // 2) * s], 30 * s, outline=ink, width=3 * s)
    else:
        a = np.array(logo)
        vis = a[a[..., 3] > 200][:, :3].astype(np.float32)
        lv = 0.299 * vis[:, 0] + 0.587 * vis[:, 1] + 0.114 * vis[:, 2] if len(vis) else np.array([128.0])
        contrast = (np.abs(lv - lum(base)) > 80).mean()
        contrast = 1 if contrast > 0.15 else contrast
        lg = _fit(logo, 460 * s, 230 * s)
        if contrast < 0.35:   # pouco contraste: selo branco atrás da logo
            pad = 34 * s
            bw, bh = lg.width + 2 * pad, lg.height + 2 * pad
            d.rounded_rectangle([lcx * s - bw // 2, lcy * s - bh // 2, lcx * s + bw // 2, lcy * s + bh // 2], 36 * s, fill="white")
        im.paste(lg, (lcx * s - lg.width // 2, lcy * s - lg.height // 2), lg)
    d = ImageDraw.Draw(im)
    if nome:
        f = font("Medium", 30)
        t = nome.upper()[:32]
        while d.textlength(t, font=f) > 800 * s and f.size > 18 * s:
            f = ImageFont.truetype(A("Poppins-Medium.ttf"), f.size - 2 * s)
        d.text((Wp / 2, 375 * s), t, font=f, fill=ink, anchor="mm")

    # arte NFC
    _nfc_art(d, 520, 570, s, ink, base)

    # Aproxime o celular
    fb = font("Bold", 46)
    fr = font("Regular", 46)
    t1, t2 = "Aproxime", " o celular"
    w1, w2 = d.textlength(t1, font=fb), d.textlength(t2, font=fr)
    x0 = Wp / 2 - (w1 + w2) / 2
    d.text((x0, 820 * s), t1, font=fb, fill=ink, anchor="lm")
    d.text((x0 + w1, 820 * s), t2, font=fr, fill=ink, anchor="lm")

    # ícones
    _icon_row(d, 935, s, ink, base)

    # QR
    qs, qy = 380, 1020
    qx = 500 - qs // 2
    d.rounded_rectangle([qx * s, qy * s, (qx + qs) * s, (qy + qs) * s], 34 * s, fill="white")
    qr = Image.open(A("qr.png")).convert("RGB").resize(((qs - 64) * s, (qs - 64) * s), Image.NEAREST)
    im.paste(qr, ((qx + 32) * s, (qy + 32) * s))
    f = font("Bold", 30)
    d.text((Wp / 2, (qy + qs + 42) * s), "Escaneie o QR Code", font=f, fill=ink, anchor="mm")

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
