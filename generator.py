"""Gera a arte da plaquinha NFC a partir de uma logo e o mockup 3D no display de acrílico."""
import io
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

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


def _fa_font(kind, size):
    """Fonte de ícones Font Awesome (instalada via pip fontawesomefree). None se não existir."""
    try:
        import fontawesomefree
        base = os.path.join(os.path.dirname(fontawesomefree.__file__), "static", "fontawesomefree", "webfonts")
        return ImageFont.truetype(os.path.join(base, f"fa-{kind}.ttf"), size)
    except Exception:
        return None


def _icon_row(d, cy, s, ink, bg):
    lw = 5 * s
    xs = [500 - 2 * 150, 500 - 150, 500, 500 + 150, 500 + 300]
    y = cy * s
    fb = _fa_font("brands-400", 66 * s)
    # Google
    x = xs[0] * s
    if fb:
        d.text((x, y), "\uf1a0", font=fb, fill=ink, anchor="mm")
    else:
        d.text((x, y + 4 * s), "G", font=ImageFont.truetype(A("Poppins-Bold.ttf"), 78 * s), fill=ink, anchor="mm")
    # Wi-Fi (selo)
    x = xs[1] * s
    d.rounded_rectangle([x - 54 * s, y - 27 * s, x + 54 * s, y + 27 * s], 27 * s, outline=ink, width=lw)
    d.text((x, y + 2 * s), "WiFi", font=ImageFont.truetype(A("Poppins-Bold.ttf"), 30 * s), fill=ink, anchor="mm")
    # Pix
    x = xs[2] * s
    if fb and fb.getbbox("\ue43a")[2] > 0:
        d.text((x, y), "\ue43a", font=fb, fill=ink, anchor="mm")
    else:
        for dx, dy in ((0, -17), (17, 0), (0, 17), (-17, 0)):
            c2x, c2y = x + dx * s, y + dy * s
            r = 13 * s
            d.polygon([(c2x, c2y - r), (c2x + r, c2y), (c2x, c2y + r), (c2x - r, c2y)], fill=ink)
    # Instagram
    x = xs[3] * s
    if fb:
        d.text((x, y), "\uf16d", font=fb, fill=ink, anchor="mm")
    else:
        d.rounded_rectangle([x - 32 * s, y - 32 * s, x + 32 * s, y + 32 * s], 16 * s, outline=ink, width=lw)
        d.ellipse([x - 14 * s, y - 14 * s, x + 14 * s, y + 14 * s], outline=ink, width=lw)
        d.ellipse([x + 15 * s, y - 23 * s, x + 23 * s, y - 15 * s], fill=ink)
    # WhatsApp
    x = xs[4] * s
    if fb:
        d.text((x, y), "\uf232", font=fb, fill=ink, anchor="mm")
    else:
        d.ellipse([x - 31 * s, y - 31 * s, x + 31 * s, y + 29 * s], outline=ink, width=lw)
        d.polygon([(x - 25 * s, y + 12 * s), (x - 34 * s, y + 36 * s), (x - 9 * s, y + 26 * s)], fill=ink)



def nfc_mask(scale=4):
    W, H = 640, 370
    k = scale
    def P(*v): return [x * k for x in v]
    M = Image.new("L", (W * k, H * k), 0)
    d = ImageDraw.Draw(M)
    # palma (forma sólida), suavizada
    palm = Image.new("L", M.size, 0); pd = ImageDraw.Draw(palm)
    pd.ellipse(P(478, 92, 556, 182), fill=255)            # polegar
    pd.polygon(P(470, 150, 560, 130, 600, 210, 612, 300, 622, 360, 470, 362, 418, 318, 470, 290), fill=255)
    pd.ellipse(P(540, 150, 615, 300), fill=255)
    palm = palm.filter(ImageFilter.GaussianBlur(6 * k)).point(lambda v: 255 if v > 128 else 0)
    M.paste(255, (0, 0), palm)
    # recorte da tela do celular + contorno
    d.rounded_rectangle(P(352, 42, 522, 302), 24 * k, fill=0)
    d.rounded_rectangle(P(352, 42, 522, 302), 24 * k, outline=255, width=14 * k)
    # oval aberto na direita
    sw = 15 * k
    box = P(40, 18, 380, 238)
    d.arc(box, 38, 322, fill=255, width=sw)
    cx, cy, rx, ry = 210 * k, 128 * k, 170 * k - sw / 2, 110 * k - sw / 2
    for a in (38, 322):
        t = np.radians(a); x, y = cx + rx * np.cos(t), cy + ry * np.sin(t)
        d.ellipse([x - sw / 2, y - sw / 2, x + sw / 2, y + sw / 2], fill=255)
    # ondas
    wx, wy = 185 * k, 128 * k
    for r in (24, 54, 84):
        R = r * k; ww = 18 * k
        d.arc([wx - R, wy - R, wx + R, wy + R], -48, 48, fill=255, width=ww)
        for a in (-48, 48):
            t = np.radians(a); rr = R - ww / 2
            x, y = wx + rr * np.cos(t), wy + rr * np.sin(t)
            d.ellipse([x - ww / 2, y - ww / 2, x + ww / 2, y + ww / 2], fill=255)
    # dedos (pontas arredondadas sobre a borda esquerda do celular)
    for fy in (96, 140, 184, 228):
        d.ellipse(P(330, fy + 2, 382, fy + 38), fill=255)
    return M.filter(ImageFilter.GaussianBlur(0.8 * k))



def _nfc_art(im, cx, cy, s, ink, bg=None, width=470):
    """Ícone 'aproxime o celular' no mesmo estilo das plaquinhas do Marcos."""
    m = nfc_mask(3)
    w = int(width * s)
    h = int(m.height * w / m.width)
    m = m.resize((w, h), Image.LANCZOS)
    im.paste(ink, (int(cx * s - w / 2), int(cy * s - h / 2)), m)


def extract_badge(raw, main):
    """Para foto/print: acha a maior mancha da cor da marca (ex.: círculo da logo),
    recorta só ela (sem o fundo) e limpa ícones escuros pequenos na borda.
    Retorna RGBA ou None."""
    rgb = np.array(raw.convert("RGB"))
    h, w = rgb.shape[:2]
    diff = np.abs(rgb.astype(np.int16) - np.array(main, np.int16)).sum(2)
    m = (diff < 110).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 0.12 * h * w:
        return None
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    fill = cv2.drawContours(np.zeros_like(comp), cnts, -1, 1, -1)
    circular = fill.sum() / (np.pi * r * r) > 0.72
    if circular:
        r *= 0.985
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    else:
        return None
    # remove ícones escuros pequenos encostados na borda (ex.: lápis do print do WhatsApp)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat = rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)
    dark = ((gray < 70) & (sat < 45) & (mask > 0)).astype(np.uint8)
    nd, ld, sd, cd = cv2.connectedComponentsWithStats(dark)
    fix = np.zeros((h, w), np.uint8)
    for j in range(1, nd):
        a = sd[j, cv2.CC_STAT_AREA]
        dist = np.hypot(cd[j][0] - cx, cd[j][1] - cy)
        if a < 0.03 * area and dist > 0.6 * r:
            fix[ld == j] = 255
    if fix.any():
        fix = cv2.dilate(fix, np.ones((7, 7), np.uint8))
        rgb = cv2.inpaint(rgb, fix, 7, cv2.INPAINT_TELEA)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    ys, xs = np.nonzero(mask > 10)
    out = np.dstack([rgb, mask])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def _tools_pattern(Wp, Hp, s):
    """Padrão de ferramentas em traço fino (estilo das folhas de milho/palmeira das plaquinhas)."""
    def outline(m, w):
        a = np.array(m)
        er = cv2.erode(a, np.ones((w, w), np.uint8))
        return Image.fromarray(((a > 127) & (er < 128)).astype(np.uint8) * 255)
    def wrench():
        m = Image.new("L", (400, 400), 0); d = ImageDraw.Draw(m)
        d.rounded_rectangle([180, 90, 220, 330], 20, fill=255)
        d.ellipse([130, 20, 270, 160], fill=255); d.ellipse([145, 290, 255, 400], fill=255)
        d.rectangle([180, 10, 220, 85], fill=0); d.ellipse([170, 315, 230, 375], fill=0)
        return m.rotate(40, resample=Image.BICUBIC)
    def hammer():
        m = Image.new("L", (400, 400), 0); d = ImageDraw.Draw(m)
        d.rounded_rectangle([182, 120, 218, 390], 16, fill=255)
        d.rounded_rectangle([90, 60, 300, 135], 14, fill=255)
        d.polygon([(300, 70), (360, 40), (340, 100), (300, 125)], fill=255)
        return m.rotate(-30, resample=Image.BICUBIC)
    def screw():
        m = Image.new("L", (400, 400), 0); d = ImageDraw.Draw(m)
        d.rounded_rectangle([130, 40, 270, 90], 18, fill=255)
        d.polygon([(165, 90), (235, 90), (235, 300), (200, 370), (165, 300)], fill=255)
        o = outline(m, 9); d2 = ImageDraw.Draw(o)
        for y in range(115, 300, 32):
            d2.line([(165, y + 14), (235, y)], fill=255, width=8)
        d2.line([(200, 50), (200, 80)], fill=255, width=8)
        return o.rotate(25, resample=Image.BICUBIC), True
    def nut():
        m = Image.new("L", (400, 400), 0); d = ImageDraw.Draw(m)
        pts = [(200 + 150 * np.cos(np.radians(a)), 200 + 150 * np.sin(np.radians(a))) for a in range(0, 360, 60)]
        d.polygon(pts, fill=255); d.ellipse([130, 130, 270, 270], fill=0)
        return m
    def gear():
        m = Image.new("L", (400, 400), 0); d = ImageDraw.Draw(m)
        d.ellipse([70, 70, 330, 330], fill=255)
        for a in range(0, 360, 40):
            im2 = Image.new("L", (400, 400), 0)
            ImageDraw.Draw(im2).rectangle([175, 20, 225, 90], fill=255)
            m.paste(255, (0, 0), im2.rotate(a, resample=Image.BICUBIC))
        d.ellipse([150, 150, 250, 250], fill=0)
        return m
    makers = [wrench, hammer, screw, nut, gear]
    # posições (x, y, tamanho, rot, tipo) — mais nas bordas, como na arte de referência
    spots = [(-30, 40, 300, 10, 0), (880, -20, 260, 0, 4), (930, 420, 230, 15, 1), (-60, 520, 250, -20, 2),
             (40, 980, 220, 30, 3), (900, 900, 280, -10, 0), (-40, 1330, 300, 0, 4), (860, 1300, 260, 20, 2),
             (650, 1450, 180, 0, 1)]
    out = Image.new("L", (Wp, Hp), 0)
    for x, y, sz, rot, t in spots:
        r = makers[t]()
        if isinstance(r, tuple):
            o = r[0]
        else:
            o = outline(r, 9)
        o = o.rotate(rot, resample=Image.BICUBIC).resize((sz * s, sz * s), Image.LANCZOS)
        out.paste(255, (x * s - sz * s // 2 + sz * s // 2 - sz * s // 2, y * s - sz * s // 2 + sz * s // 2 - sz * s // 2), o)
    return out


def _dominant(px):
    q = (px // 16).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    b = np.bincount(key).argmax()
    return tuple(int(v) for v in px[key == b].mean(0))


def _key_out(rgb, bg, base_alpha=None, t0=20, t1=60):
    """Remove a cor de fundo com borda suave e 'descontamina' a cor das bordas."""
    f = rgb.astype(np.float32)
    B = np.array(bg, np.float32)
    dist = np.sqrt(((f - B) ** 2).sum(2))
    a = np.clip((dist - t0) / (t1 - t0), 0, 1)
    if base_alpha is not None:
        a = a * (base_alpha.astype(np.float32) / 255)
    aa = np.maximum(a, 1e-3)[..., None]
    F = np.clip((f - (1 - aa) * B) / aa, 0, 255)
    out = np.dstack([F, a * 255]).astype(np.uint8)
    on = (out[..., 3] > 60).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(on, np.ones((5, 5), np.uint8)))
    if n > 2:
        tot = st[1:, cv2.CC_STAT_AREA].sum()
        bigx = st[1:, cv2.CC_STAT_AREA].max()
        for j in range(1, n):
            ar_ = st[j, cv2.CC_STAT_AREA]
            # risquinho/ícone solto (ex.: lápis do print) bem menor que o resto
            if ar_ < 0.0025 * tot or (ar_ < 0.04 * tot and st[j, cv2.CC_STAT_HEIGHT] > 3 * st[j, cv2.CC_STAT_WIDTH] and ar_ < 0.1 * bigx and False):
                out[..., 3][lab == j] = 0
    ys, xs = np.nonzero(out[..., 3] > 25)
    if len(xs) == 0:
        return None
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


def integrate_logo(raw, main):
    """Tira o fundo da logo para ela ficar 'impressa' direto na plaquinha.
    Retorna (logo_RGBA, cor_do_fundo_original|None, tipo)."""
    a = np.array(raw.convert("RGBA"))
    if (a[..., 3] < 250).mean() > 0.02:                # já é PNG sem fundo
        lg, _ = clean_logo(raw)
        return lg, None, "png"
    rgb = a[..., :3]
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.int16)
    bmed = np.median(border, axis=0)
    flat = np.abs(border - bmed).sum(1).mean() <= 45
    def badge_key(img):
        badge = extract_badge(img, main)
        if badge is None:
            return None
        ba = np.array(badge)
        inside = ba[..., :3][ba[..., 3] > 200]
        bg = _dominant(inside)
        frac = (np.sqrt(((inside.astype(np.float32) - bg) ** 2).sum(1)) < 30).mean()
        if frac > 0.35:
            m = cv2.erode(ba[..., 3].copy(), np.ones((9, 9), np.uint8))
            # tira restos pequenos perto da borda do círculo (ícone de lápis, sombra etc.)
            dd = np.sqrt(((ba[..., :3].astype(np.float32) - bg) ** 2).sum(2))
            fg = ((dd > 40) & (m > 128)).astype(np.uint8)
            n, lab, st, ce = cv2.connectedComponentsWithStats(cv2.dilate(fg, np.ones((5, 5), np.uint8)))
            hh, ww = m.shape
            R = min(hh, ww) / 2
            tot = max(1, fg.sum())
            for j in range(1, n):
                if st[j, cv2.CC_STAT_AREA] < 0.06 * tot and np.hypot(ce[j][0] - ww / 2, ce[j][1] - hh / 2) > 0.62 * R:
                    cpx = ba[..., :3][(lab == j) & (fg > 0)].astype(np.int16)
                    if len(cpx) and (cpx.max(1) - cpx.min(1)).mean() < 50 and cpx.mean() < 110:   # só ícone cinza/escuro
                        m[lab == j] = 0
            lg = _key_out(ba[..., :3], bg, m)
            if lg is not None:
                return lg, bg, "key"
        return badge, None, "badge"

    # 1) fundo liso (branco, preto ou colorido)
    if flat:
        bg = tuple(int(v) for v in bmed)
        lg = _key_out(rgb, bg)
        if lg is not None:
            # sobrou só um círculo de cor única (foto de perfil)? então a plaquinha vira a cor do círculo
            la = np.array(lg)[..., 3]
            cnts, _ = cv2.findContours((la > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                (_, _), r = cv2.minEnclosingCircle(c)
                if sum(1 for o in cnts if o is not c and cv2.contourArea(o) > 0.004 * cv2.contourArea(c)) == 0 and cv2.contourArea(c) / (np.pi * r * r) > 0.8:
                    bk = badge_key(raw)
                    if bk is not None and bk[2] == "key":
                        return bk
            return lg, bg, "key"
    # 2) print/foto com logo redonda
    bk = badge_key(raw)
    if bk is not None:
        return bk
    # 3) foto
    return raw, None, "photo"


def make_design(logo_src, nome=None):
    raw = load_image(logo_src)
    logo, complex_bg = clean_logo(raw)
    rgba = np.array(raw)
    opaque = (rgba[..., 3] > 250).mean() > 0.98
    ra0 = rgba[..., :3].astype(np.float32)
    border = np.concatenate([ra0[0], ra0[-1], ra0[:, 0], ra0[:, -1]])
    bmed = tuple(int(v) for v in np.median(border, axis=0))
    main, sec = brand_colors(logo if not complex_bg else raw)
    if lum(main) > 225 and (max(main) - min(main)) < 40:
        main, sec = (sec or (40, 40, 40)), None
    vivid = (max(main) - min(main)) > 60
    s = SS
    Wp, Hp = W * s, H * s

    # ---- cor da plaquinha = cor do fundo da logo (assim a logo fica integrada)
    lg_int, lbg, lkind = integrate_logo(raw, main)
    if lbg is not None and lum(lbg) > 215 and (max(lbg) - min(lbg)) < 30:
        base = mix(lbg, main, 0.07) if vivid else lbg       # logo em fundo branco: plaquinha clara
    elif lbg is not None:
        base = lbg                                          # fundo colorido/escuro: plaquinha na mesma cor
    elif vivid and lum(main) > 150:
        base = mix((14, 14, 16), main, 0.06)
    elif vivid:
        base = shade(main, 0.62)
    else:
        base = mix(main, (18, 18, 20), 0.6)
    dark = lum(base) < 150
    ink = (255, 255, 255) if dark else (32, 32, 38)
    accent = main if vivid else (sec or ink)
    if abs(lum(accent) - lum(base)) < 45:   # destaque precisa aparecer no fundo
        accent = sec if (sec and abs(lum(sec) - lum(base)) >= 45) else ink

    # fundo cor da marca (chapado, degradê bem leve) + padrão de ferramentas em traço
    top = mix(base, (255, 255, 255), 0.06) if dark else base
    bot = shade(base, 0.82) if dark else mix(base, (0, 0, 0), 0.05)
    g = np.linspace(0, 1, Hp)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - g) + np.array(bot)[None, None, :] * g).repeat(Wp, axis=1)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    # círculos decorativos nos cantos (igual às primeiras prévias)
    light = lum(base) > 200
    if light:
        c1 = main if vivid else mix(main, (120, 120, 120), 0.3)
        c2 = sec or mix(c1, (255, 255, 255), 0.45)
    else:
        c1 = sec if (sec and abs(lum(sec) - lum(base)) > 40) else (shade(base, 0.72) if lum(base) > 90 else mix(base, (255, 255, 255), 0.18))
        c2 = mix(c1, base, 0.5)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    def circ(cx, cy, r, col, a):
        ld.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=col + (a,))
    circ(1010, 20, 300, c2, 150)
    circ(960, -40, 250, c1, 255)
    circ(-10, 1480, 250, c2, 150)
    circ(40, 1530, 210, c1, 255)
    im = Image.alpha_composite(im, layer).convert("RGB")
    d = ImageDraw.Draw(im)

    # ---- logo integrada (sem fundo)
    lcx, lcy = 500, 210
    badge = lg_int if lkind == "badge" else None

    def shadow(box, radius, ellipse=False):
        nonlocal im
        sh = Image.new("L", (im.width // 4, im.height // 4), 0)
        sd = ImageDraw.Draw(sh)
        b = [v / 4 for v in box]
        b[1] += 6 * s / 4; b[3] += 10 * s / 4
        (sd.ellipse(b, fill=110) if ellipse else sd.rounded_rectangle(b, radius / 4, fill=110))
        sh = sh.filter(ImageFilter.GaussianBlur(14 * s / 4)).resize(im.size, Image.BILINEAR)
        im.paste((0, 0, 0), (0, 0), sh)

    if lkind == "key":
        ar = lg_int.width / lg_int.height
        bw, bh = (480, 270) if ar > 1.3 else (340, 290)
        lg = _fit(lg_int, bw * s, bh * s)
        im.paste(lg, (lcx * s - lg.width // 2, lcy * s - lg.height // 2), lg)
    elif badge is not None:
        dia = 300
        lg = _fit(badge, dia * s, dia * s)
        box = [lcx * s - lg.width // 2, lcy * s - lg.height // 2, lcx * s + lg.width // 2, lcy * s + lg.height // 2]
        shadow(box, 0, ellipse=True)
        ring = 8 * s
        ImageDraw.Draw(im).ellipse([box[0] - ring, box[1] - ring, box[2] + ring, box[3] + ring], fill=(255, 255, 255))
        im.paste(lg, (box[0], box[1]), lg)
    elif lkind == "photo":
        ar = raw.width / raw.height
        if 0.8 <= ar <= 1.25:
            dia = 300
            ph = _fit(raw.convert("RGB"), dia * s, dia * s, cover=True)
            box = [(lcx - dia // 2) * s, (lcy - dia // 2) * s, (lcx + dia // 2) * s, (lcy + dia // 2) * s]
            shadow(box, 0, ellipse=True)
            ring = 8 * s
            ImageDraw.Draw(im).ellipse([box[0] - ring, box[1] - ring, box[2] + ring, box[3] + ring], fill=(255, 255, 255))
            m = Image.new("L", ph.size, 0)
            ImageDraw.Draw(m).ellipse([0, 0, ph.width - 1, ph.height - 1], fill=255)
            im.paste(ph, (box[0], box[1]), m)
        else:
            cw, ch = 560, 280
            ph = _fit(raw.convert("RGB"), cw * s, ch * s, cover=True)
            box = [(lcx - cw // 2) * s, (lcy - ch // 2) * s, (lcx + cw // 2) * s, (lcy + ch // 2) * s]
            shadow(box, 30 * s)
            ImageDraw.Draw(im).rounded_rectangle([box[0] - 7 * s, box[1] - 7 * s, box[2] + 7 * s, box[3] + 7 * s], 36 * s, fill=(255, 255, 255))
            m = Image.new("L", ph.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 30 * s, fill=255)
            im.paste(ph, (box[0], box[1]), m)
    else:
        lg = _fit(lg_int, 420 * s, 240 * s)
        if False:  # logo em cartão branco arredondado (fica limpa e legível)
            pw, ph_ = lg.width + 90 * s, lg.height + 70 * s
            box = [lcx * s - pw // 2, lcy * s - ph_ // 2, lcx * s + pw // 2, lcy * s + ph_ // 2]
            shadow(box, 40 * s)
            ImageDraw.Draw(im).rounded_rectangle(box, 40 * s, fill=(255, 255, 255))
        im.paste(lg, (lcx * s - lg.width // 2, lcy * s - lg.height // 2), lg)
    d = ImageDraw.Draw(im)

    # arte NFC
    d = ImageDraw.Draw(im)
    d.ellipse([(500 - 165) * s, (585 - 165) * s, (500 + 165) * s, (585 + 165) * s], outline=ink, width=6 * s)
    _nfc_art(im, 500, 590, s, ink, width=270)
    d = ImageDraw.Draw(im)

    # Aproxime o celular
    fb = font("Bold", 46)
    fr = font("Regular", 46)
    t1, t2 = "Aproxime", " o celular"
    w1, w2 = d.textlength(t1, font=fb), d.textlength(t2, font=fr)
    x0 = Wp / 2 - (w1 + w2) / 2
    d.text((x0, 830 * s), t1, font=fb, fill=ink, anchor="lm")
    d.text((x0 + w1, 830 * s), t2, font=fr, fill=ink, anchor="lm")

    _icon_row(d, 935, s, ink, base)
    fl = font("Medium", 22)
    for i, lab in enumerate(["Avalie", "Wi-Fi", "Pix", "Instagram", "WhatsApp"]):
        d.text(((200 + 150 * i) * s, 995 * s), lab, font=fl, fill=ink, anchor="mm")

    # QR
    qs, qy = 340, 1040
    qx = 500 - qs // 2
    d.rounded_rectangle([qx * s, qy * s, (qx + qs) * s, (qy + qs) * s], 34 * s, fill="white", outline=(None if dark else ink), width=(0 if dark else 3 * s))
    qr = Image.open(A("qr.png")).convert("RGB").resize(((qs - 64) * s, (qs - 64) * s), Image.NEAREST)
    im.paste(qr, ((qx + 32) * s, (qy + 32) * s))
    f = font("Bold", 30)
    d.text((Wp / 2, (qy + qs + 40) * s), "Escaneie o QR Code", font=f, fill=ink, anchor="mm")

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
