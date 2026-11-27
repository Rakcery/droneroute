"""
Y-3 · Algılama hata analizi: model NEREDE ve NEDEN yanılıyor?

Ne yapar:
  1. Test setindeki (bolme.json -> "test") her resimde modelin tahminlerini alır (ya da --tahmin klasöründen okur).
  2. Her gerçek nesneyi (etiket) aynı sınıftan, kutusu en çok örtüşen tahminle eşler (IoU >= 0.5 -> "bulundu").
     Eşleşmeyen gerçek nesne = KAÇIRILDI. Eşleşmeyen tahmin = YANLIŞ ALARM.
  3. Her nesne için olası sebepleri ölçer: boyu (piksel), uzaklığı (eğik kamerada resmin üstü = uzak),
     bulanıklığı, kadrajdan kesik olması. Başarı oranını bunlara göre tablolar.
  4. Enes'e özet: "şu boydan küçük nesneleri güvenilir bulamıyoruz" + önerilen güvenlik payı (veri/algi_ozet.json, S-6).
  5. Kaçırılan ve yanlış alarm örneklerini kırpıp veri/_hata_ornekleri/ altına kaydeder (gözle incelemek için).

Nesne nesne (IoU ile) analiz sadece ARAÇ ve İNSAN için anlamlı: UAVid bitişik binaları ve ağaç şeritlerini tek
bölge etiketliyor, model ise tek tek bina/ağaç buluyor; kutu eşleme orada yanıltır. Bina/ağaç için alan bazlı
sonuç Y-2'de (sonuclar/y-2_ince_ayar.csv).

Çalıştır (depo kökünden):
  python tools/hata_analizi.py                               # models/kendi_y2.pt ile test seti (CPU'da birkaç dk)
  python tools/hata_analizi.py --tahmin veri/_tahminler       # daha önce kaydedilmiş tahminlerle (hızlı)
  python tools/hata_analizi.py --deneme                       # sahte veriyle kendi kendini dene (model gerekmez)
Çıktılar: sonuclar/y-3_hata.csv (S-6), veri/algi_ozet.json (S-6, Enes'e), veri/_hata_ornekleri/*.jpg (depoya girmez),
          veri/_tahminler/ (modelin tahminleri, depoya girmez). Ekrana Markdown tablolar (veri/hata_analizi.md için).
"""
import argparse                              # komut satırı seçenekleri
import json                                  # bolme.json, algi_ozet.json
import math                                  # ceil, sqrt
import os                                    # dosya yolları
import sys                                   # depo kökünü import yoluna eklemek için
import tempfile                              # --deneme için geçici klasör
from collections import defaultdict          # sayaçlar

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # depo kökü
sys.path.insert(0, KOK)

import cv2                                   # resim okuma, bulanıklık ölçümü, kırpma
import numpy as np                           # sayı tabloları

import sahte_veri                            # SINIFLAR, sonuc_yaz (S-6)

SINIFLAR = sahte_veri.SINIFLAR               # ['vinc', 'bina', 'agac', 'arac', 'insan', 'kus']
NESNE_SINIFLARI = (3, 4)                     # nesne nesne analiz: arac, insan (bina/agac bölge olarak etiketli)
IOU_ESIK = 0.5                               # bundan fazla örtüşüyorsa "aynı nesneyi bulmuş"
ARAC_BOYU_M = 4.5                            # ölçek cetveli: ortalama bir araç ~4.5 m (piksel -> metre için)
HUCRE_M = 2.0                                # Gazebo / eğik kamerada bir hücre (S-0: hucre_m)
BOY_ARALIKLARI = [0, 16, 24, 32, 48, 64, 96, 10_000]   # nesne boyu (√alan, 4K piksel) aralıkları
GUVENILIR = 0.5                              # bir boy aralığında en az bu kadarı bulunuyorsa "güvenilir"


# ---------------- Temel: iki kutunun örtüşmesi (kendi kodumuz) ----------------
def iou(a, b):
    """İki kutunun (x1, y1, x2, y2) örtüşmesi: ortak alan / toplam kaplanan alan. 1 = birebir, 0 = hiç değmiyor."""
    gx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))   # kesişimin genişliği (değmiyorsa 0)
    gy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))   # kesişimin yüksekliği
    kes = gx * gy                                       # ortak alan
    alan = lambda k: (k[2] - k[0]) * (k[3] - k[1])      # bir kutunun alanı
    birlesim = alan(a) + alan(b) - kes                  # ortak kısmı bir kez say
    return kes / birlesim if birlesim > 0 else 0.0


# ---------------- Etiket / tahmin okuma ----------------
def etiket_oku(yol, W, H, guven_var=False):
    """
    S-5 etiket dosyası -> [(sinif, kutu(x1,y1,x2,y2) piksel, guven)]. Satır kutu (5 sayı) ya da çokgen olabilir;
    tahmin dosyalarında sonda bir güven sayısı daha var (guven_var=True).
    """
    out = []
    if not os.path.exists(yol):
        return out
    for satir in open(yol, encoding="utf-8"):
        p = satir.split()
        if len(p) < 5:
            continue
        sinif, deger = int(p[0]), [float(v) for v in p[1:]]
        guven = 1.0
        if guven_var:
            guven, deger = deger[-1], deger[:-1]
        if len(deger) == 4:                  # kutu: xc yc w h
            xc, yc, w, h = deger
            kutu = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
        else:                                # çokgen: noktaların kapsayan kutusu
            a = np.array(deger).reshape(-1, 2) * [W, H]
            kutu = (*a.min(0), *a.max(0))
        out.append((sinif, tuple(float(v) for v in kutu), guven))
    return out


def model_tahminleri(model, resim_yolu, imgsz, kayit_yolu):
    """Modelin tahminlerini alır, S-5 kutu biçiminde (+güven) kaydeder: bir dahaki sefere --tahmin ile hızlı."""
    r = model.predict(resim_yolu, imgsz=imgsz, conf=0.25, verbose=False)[0]
    H, W = r.orig_shape
    satirlar = []
    for (x1, y1, x2, y2), k, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        satirlar.append(f"{int(k)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f} {c:.4f}")
    os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
    with open(kayit_yolu, "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar) + "\n")


# ---------------- Eşleme ----------------
def esle(gercekler, tahminler):
    """
    Aynı sınıftaki gerçek ve tahmin kutularını açgözlü eşler: en yüksek IoU'lu çift önce. Dönen:
    eslesen [(g_indeks, t_indeks, iou)], kacan [g_indeks], fazla [t_indeks].
    """
    adaylar = []
    for gi, (gs, gk, _) in enumerate(gercekler):
        for ti, (ts, tk, _) in enumerate(tahminler):
            if gs == ts:
                o = iou(gk, tk)
                if o >= IOU_ESIK:
                    adaylar.append((o, gi, ti))
    adaylar.sort(reverse=True)               # en iyi örtüşen çift önce
    g_kullanildi, t_kullanildi, eslesen = set(), set(), []
    for o, gi, ti in adaylar:
        if gi not in g_kullanildi and ti not in t_kullanildi:
            g_kullanildi.add(gi); t_kullanildi.add(ti); eslesen.append((gi, ti, o))
    kacan = [i for i in range(len(gercekler)) if i not in g_kullanildi]
    fazla = [i for i in range(len(tahminler)) if i not in t_kullanildi]
    return eslesen, kacan, fazla


# ---------------- Sebep ölçüleri ----------------
def boyut_px(kutu):
    """Nesnenin boyu: kutu alanının karekökü (piksel). Kare bir nesnenin kenar uzunluğu gibi düşün."""
    return math.sqrt(max(kutu[2] - kutu[0], 1) * max(kutu[3] - kutu[1], 1))


def uzaklik_bandi(kutu, H):
    """Eğik kamerada resmin üstü uzak, altı yakın. Nesnenin alt kenarı (yere değdiği yer) hangi üçte birde?"""
    y = kutu[3] / H
    return "uzak" if y < 1 / 3 else ("orta" if y < 2 / 3 else "yakin")


PENCERE = 96                                 # keskinlik ölçüm penceresi (piksel), nesnenin boyundan bağımsız


def bulaniklik(gri, kutu):
    """
    Nesnenin merkezindeki SABİT boy (96x96) pencerede Laplacian varyansı: düşükse kenarlar yumuşak = bulanık.
    Nesnenin kendi kutusunda ölçseydik minik kutular yapay olarak "keskin" çıkardı (ilk denemede oldu): ölçü boyu ölçerdi.
    """
    cx, cy = int((kutu[0] + kutu[2]) / 2), int((kutu[1] + kutu[3]) / 2)
    H, W = gri.shape
    x1, y1 = min(max(cx - PENCERE // 2, 0), W - PENCERE), min(max(cy - PENCERE // 2, 0), H - PENCERE)
    k = gri[y1:y1 + PENCERE, x1:x1 + PENCERE]
    return float(cv2.Laplacian(k, cv2.CV_64F).var())


def kesik_mi(kutu, W, H, pay=2):
    """Kutu resmin kenarına değiyor mu? (nesnenin bir kısmı kadraj dışında olabilir)"""
    return kutu[0] <= pay or kutu[1] <= pay or kutu[2] >= W - pay or kutu[3] >= H - pay


def metre_olcegi(gercekler, W, H):
    """
    Piksel -> metre cetveli: aynı resimdeki araçların uzun kenarı ~4.5 m. Eğik kamerada ölçek uzaklıkla değiştiği
    için her uzaklık bandına ayrı ölçek (o banttaki araçların medyanı); bantta araç yoksa resmin tamamının medyanı.
    Yaklaşık bir değer: kesin ölçek için kamera kalibrasyonu lazım (Y-5).
    """
    bant = defaultdict(list)
    for s, k, _ in gercekler:
        if s == 3:
            bant[uzaklik_bandi(k, H)].append(max(k[2] - k[0], k[3] - k[1]))
    hepsi = [v for b in bant.values() for v in b]
    if not hepsi:
        return {}
    genel = ARAC_BOYU_M / float(np.median(hepsi))
    return {b: (ARAC_BOYU_M / float(np.median(bant[b])) if len(bant[b]) >= 3 else genel) for b in ("uzak", "orta", "yakin")}


# ---------------- Ana analiz ----------------
def test_resimleri(veri_kok):
    """bolme.json'daki test sahnelerinin resimleri."""
    with open(os.path.join(veri_kok, "bolme.json"), encoding="utf-8") as f:
        bolme = json.load(f)
    out = []
    for sahne in bolme["test"]:
        klasor = os.path.join(veri_kok, sahne, "images")
        out += [(sahne, os.path.join(klasor, d)) for d in sorted(os.listdir(klasor)) if d.lower().endswith((".jpg", ".png"))]
    return out


def analiz(veri_kok, tahmin_kok, model=None, imgsz=1280, ornek_kok=None, en_fazla_ornek=30):
    """Bütün test setini analiz eder; nesne kayıtları ve özet sayılar döner."""
    kayitlar = []                            # her gerçek nesne ve her yanlış alarm için bir satır
    hata_hata = []                           # (puan, kırpma bilgisi) örnek seçmek için
    for sahne, resim in test_resimleri(veri_kok):
        ad = os.path.splitext(os.path.basename(resim))[0]
        tahmin_yolu = os.path.join(tahmin_kok, sahne, ad + ".txt")
        if model is not None:
            model_tahminleri(model, resim, imgsz, tahmin_yolu)
        img = cv2.imread(resim)
        H, W = img.shape[:2]
        gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        etiket_yolu = resim.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        gercek = [g for g in etiket_oku(etiket_yolu, W, H) if g[0] in NESNE_SINIFLARI]
        tahmin = [t for t in etiket_oku(tahmin_yolu, W, H, guven_var=True) if t[0] in NESNE_SINIFLARI]
        olcek = metre_olcegi(gercek, W, H)
        eslesen, kacan, fazla = esle(gercek, tahmin)
        eslesme = {gi: (ti, o) for gi, ti, o in eslesen}
        for gi, (s, k, _) in enumerate(gercek):
            bant = uzaklik_bandi(k, H)
            m_px = olcek.get(bant)
            kayit = dict(sahne=sahne, resim=ad, sinif=SINIFLAR[s], durum="bulundu" if gi in eslesme else "kacirildi",
                         boy_px=boyut_px(k), bant=bant, bulaniklik=bulaniklik(gri, k), kesik=kesik_mi(k, W, H),
                         boy_m=boyut_px(k) * m_px if m_px else None, kenar_hatasi_m=None, kutu=k)
            if gi in eslesme:                # konum hatası: tahmin kutusunun kenarı gerçeğinden en çok ne kadar kaymış
                tk = tahmin[eslesme[gi][0]][1]
                hata_px = max(abs(a - b) for a, b in zip(k, tk))
                kayit["kenar_hatasi_m"] = hata_px * m_px if m_px else None
            kayitlar.append(kayit)
        for ti in fazla:                     # yanlış alarm: başka sınıftan bir gerçeğe mi denk geliyor?
            s, k, c = tahmin[ti]
            tum_gercek = etiket_oku(etiket_yolu, W, H)
            baska = [SINIFLAR[gs] for gs, gk, _ in tum_gercek if gs != s and iou(gk, k) >= 0.3]
            kayitlar.append(dict(sahne=sahne, resim=ad, sinif=SINIFLAR[s], durum="yanlis_alarm", boy_px=boyut_px(k),
                                 bant=uzaklik_bandi(k, H), bulaniklik=bulaniklik(gri, k), kesik=kesik_mi(k, W, H),
                                 boy_m=None, kenar_hatasi_m=None, kutu=k, guven=c,
                                 aslinda=baska[0] if baska else "-"))
        if ornek_kok:                        # bu resmin hata örneklerini kırp
            for kr in kayitlar[-(len(gercek) + len(fazla)):]:
                if kr["durum"] != "bulundu":
                    hata_hata.append((kr, img))
    if ornek_kok:
        ornek_kaydet(hata_hata, ornek_kok, en_fazla_ornek)
    return kayitlar


def ornek_kaydet(hatalar, klasor, en_fazla):
    """Kaçırılan ve yanlış alarm örneklerinden çeşitli bir seçki kırpar (bağlamla birlikte, 3 kat büyütülmüş)."""
    os.makedirs(klasor, exist_ok=True)
    rng = np.random.default_rng(0)           # hep aynı örnekler
    gruplar = defaultdict(list)
    for kr, img in hatalar:
        gruplar[(kr["durum"], kr["sinif"])].append((kr, img))
    secilen = []
    for anahtar in sorted(gruplar):          # her (durum, sınıf) grubundan eşit pay
        g = gruplar[anahtar]
        n = min(len(g), max(1, en_fazla // max(len(gruplar), 1)))
        secilen += [g[i] for i in rng.choice(len(g), n, replace=False)]
    for i, (kr, img) in enumerate(secilen):
        x1, y1, x2, y2 = kr["kutu"]
        b = max(x2 - x1, y2 - y1, 40)        # nesnenin etrafında bağlam bırak
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        H, W = img.shape[:2]
        a, c = int(max(cx - 1.5 * b, 0)), int(max(cy - 1.5 * b, 0))
        kes = img[c:int(min(cy + 1.5 * b, H)), a:int(min(cx + 1.5 * b, W))].copy()
        renk = (0, 0, 255) if kr["durum"] == "kacirildi" else (0, 200, 255)
        cv2.rectangle(kes, (int(x1 - a), int(y1 - c)), (int(x2 - a), int(y2 - c)), renk, 2)
        kes = cv2.resize(kes, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) if max(kes.shape[:2]) < 200 else kes
        ad = f"{i:02d}_{kr['durum']}_{kr['sinif']}_{int(kr['boy_px'])}px_{kr['bant']}_{kr['sahne'].replace('/', '-')}_{kr['resim']}.jpg"
        cv2.imwrite(os.path.join(klasor, ad), kes)


# ---------------- Özet ----------------
def oran(liste):
    """Bulunma oranı ve adet: (bulunan / toplam, toplam)."""
    return (sum(1 for k in liste if k["durum"] == "bulundu") / len(liste) if liste else float("nan"), len(liste))


def ozet(kayitlar, model_adi):
    gercek = [k for k in kayitlar if k["durum"] != "yanlis_alarm"]
    yanlis = [k for k in kayitlar if k["durum"] == "yanlis_alarm"]
    satirlar, md = [], []

    def ekle(deney, yontem, sahne, metrik, deger, birim):
        satirlar.append(dict(kart="Y-3", deney=deney, yontem=yontem, sahne=sahne, metrik=metrik,
                             deger=round(deger, 4) if isinstance(deger, float) else deger, birim=birim))

    # 1) Sınıf başına
    md.append("| Sınıf | Gerçek nesne | Bulundu | Duyarlılık | Yanlış alarm | Kesinlik |\n|---|---|---|---|---|---|")
    for s in NESNE_SINIFLARI:
        ad = SINIFLAR[s]
        g = [k for k in gercek if k["sinif"] == ad]
        r, n = oran(g)
        fa = sum(1 for k in yanlis if k["sinif"] == ad)
        bulunan = int(round(r * n)) if n else 0
        kes = bulunan / (bulunan + fa) if bulunan + fa else float("nan")
        md.append(f"| {ad} | {n} | {bulunan} | {r:.2f} | {fa} | {kes:.2f} |")
        ekle("nesne", model_adi, ad, "duyarlilik", r, "oran"); ekle("nesne", model_adi, ad, "kesinlik", kes, "oran")

    # 2) Boya göre (asıl soru: kaç pikselden küçüğü bulunamıyor?)
    md.append("\n| Boy (4K piksel, √alan) | Modelin gözünde (1280) | araç | insan |\n|---|---|---|---|")
    en_kucuk = None
    for a, b in zip(BOY_ARALIKLARI[:-1], BOY_ARALIKLARI[1:]):
        hucre = []
        hepsi = [k for k in gercek if a <= k["boy_px"] < b]
        for s in NESNE_SINIFLARI:
            g = [k for k in hepsi if k["sinif"] == SINIFLAR[s]]
            r, n = oran(g)
            hucre.append(f"{r:.2f} ({n})" if n else "-")
            if n:
                ekle("boy", model_adi, f"{SINIFLAR[s]}_{a}-{b}px", "duyarlilik", r, "oran")
        r_hepsi, n_hepsi = oran(hepsi)
        if en_kucuk is None and n_hepsi >= 5 and r_hepsi >= GUVENILIR:
            en_kucuk = a                     # bu aralıktan itibaren nesnelerin en az yarısı bulunuyor
        etiket = f"{a}-{b}" if b < 10_000 else f"{a}+"
        md.append(f"| {etiket} | {a / 3:.0f}-{b / 3:.0f} px | {hucre[0]} | {hucre[1]} |" if b < 10_000
                  else f"| {etiket} | {a / 3:.0f}+ px | {hucre[0]} | {hucre[1]} |")

    # 3) Uzaklık, bulanıklık, kadrajdan kesik
    md.append("\n| Etken | Grup | Duyarlılık (araç + insan) |\n|---|---|---|")
    for bant in ("yakin", "orta", "uzak"):
        r, n = oran([k for k in gercek if k["bant"] == bant])
        md.append(f"| uzaklık | {bant} | {r:.2f} ({n}) |"); ekle("uzaklik", model_adi, bant, "duyarlilik", r, "oran")
    if gercek:
        esik = np.percentile([k["bulaniklik"] for k in gercek], [33, 67])
        for ad, f in (("bulanık", lambda v: v < esik[0]), ("orta", lambda v: esik[0] <= v < esik[1]), ("keskin", lambda v: v >= esik[1])):
            r, n = oran([k for k in gercek if f(k["bulaniklik"])])
            md.append(f"| keskinlik | {ad} | {r:.2f} ({n}) |"); ekle("keskinlik", model_adi, ad, "duyarlilik", r, "oran")
    for ad, deger in (("kesik", True), ("tam", False)):
        r, n = oran([k for k in gercek if k["kesik"] == deger])
        md.append(f"| kadraj | {ad} | {r:.2f} ({n}) |"); ekle("kadraj", model_adi, ad, "duyarlilik", r, "oran")

    # 4) Yanlış alarmlar aslında neydi?
    md.append("\n| Yanlış alarm (sınıf) | Aslında başka sınıf | Etiketsiz bir şey |\n|---|---|---|")
    for s in NESNE_SINIFLARI:
        y = [k for k in yanlis if k["sinif"] == SINIFLAR[s]]
        baska = defaultdict(int)
        for k in y:
            if k["aslinda"] != "-":
                baska[k["aslinda"]] += 1
        md.append(f"| {SINIFLAR[s]} ({len(y)}) | {', '.join(f'{a} {n}' for a, n in sorted(baska.items())) or '-'} | {len(y) - sum(baska.values())} |")

    # 5) Enes'e: en küçük güvenilir boy ve güvenlik payı
    boy_m = [k["boy_m"] for k in gercek if k["boy_m"] is not None and en_kucuk is not None and en_kucuk <= k["boy_px"] < en_kucuk + 16]
    hata_m = [k["kenar_hatasi_m"] for k in gercek if k["kenar_hatasi_m"] is not None]
    en_kucuk_m = float(np.median(boy_m)) if boy_m else None
    p95 = float(np.percentile(hata_m, 95)) if hata_m else None
    pay = max(1, math.ceil(p95 / HUCRE_M)) if p95 else 1
    return satirlar, md, dict(en_kucuk_px=en_kucuk, en_kucuk_m=en_kucuk_m, kenar_hatasi_p95_m=p95, pay=pay,
                              kenar_hatasi_medyan_m=float(np.median(hata_m)) if hata_m else None)


# ---------------- Kendi kendini dene ----------------
def deneme():
    """
    Sahte veri: sahte_etiket_klasoru ile "gerçek" etiketler; "tahmin" = gerçeklerin bir kısmı silinmiş + kaydırılmış
    + bir sahte yanlış alarm. Bilinen sayılar çıkmalı.
    """
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9          # notun örneği: 0.333
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0                     # hiç değmiyor
    with tempfile.TemporaryDirectory() as kok:
        sahne = os.path.join(kok, "sahte", "s1")
        sahte_veri.sahte_etiket_klasoru(sahne, n=4, seed=3)
        with open(os.path.join(kok, "bolme.json"), "w") as f:
            json.dump({"egitim": [], "dogrulama": [], "test": ["sahte/s1"]}, f)
        tahmin_kok = os.path.join(kok, "_tahminler")
        beklenen_kacan = beklenen_fa = toplam = 0
        for d in sorted(os.listdir(os.path.join(sahne, "labels"))):
            satirlar = [s.split() for s in open(os.path.join(sahne, "labels", d)) if s.strip()]
            satirlar = [s for s in satirlar if int(s[0]) in NESNE_SINIFLARI]
            toplam += len(satirlar)
            tut = satirlar[::2]               # her ikinciyi "kaçır"
            beklenen_kacan += len(satirlar) - len(tut)
            out = [f"{s[0]} {float(s[1]) + 0.002:.6f} {s[2]} {s[3]} {s[4]} 0.9" for s in tut]   # biraz kaydır
            out.append("4 0.95 0.95 0.02 0.02 0.5")   # köşede sahte bir insan: yanlış alarm
            beklenen_fa += 1
            os.makedirs(os.path.join(tahmin_kok, "sahte", "s1"), exist_ok=True)
            with open(os.path.join(tahmin_kok, "sahte", "s1", d), "w") as f:
                f.write("\n".join(out) + "\n")
        kayit = analiz(kok, tahmin_kok)
        kacan = sum(1 for k in kayit if k["durum"] == "kacirildi")
        fa = sum(1 for k in kayit if k["durum"] == "yanlis_alarm")
        assert kacan == beklenen_kacan and fa == beklenen_fa, (kacan, beklenen_kacan, fa, beklenen_fa)
        print(f"deneme: {toplam} nesne, {kacan} kaçırma ve {fa} yanlış alarm doğru sayıldı; IoU notun örneğiyle aynı (0.333)")


def main():
    ap = argparse.ArgumentParser(description="Y-3: algılama hata analizi")
    ap.add_argument("--model", default=os.path.join(KOK, "models", "kendi_y2.pt"), help="YOLO modeli")
    ap.add_argument("--tahmin", default=None, help="kaydedilmiş tahmin klasörü (verilirse model çalıştırılmaz)")
    ap.add_argument("--imgsz", type=int, default=1280, help="modelin resim boyu (eğitimdeki gibi 1280)")
    ap.add_argument("--veri", default=os.path.join(KOK, "veri"), help="veri kökü (bolme.json burada)")
    ap.add_argument("--deneme", action="store_true", help="sahte veriyle kendi kendini dene")
    args = ap.parse_args()
    if args.deneme:
        deneme(); print("TAMAM"); return
    tahmin_kok = args.tahmin or os.path.join(args.veri, "_tahminler")
    model = None
    if args.tahmin is None:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"model: {args.model} (imgsz {args.imgsz}); test resimleri işleniyor...")
    kayit = analiz(args.veri, tahmin_kok, model, args.imgsz, ornek_kok=os.path.join(args.veri, "_hata_ornekleri"))
    model_adi = os.path.splitext(os.path.basename(args.model))[0]
    satirlar, md, o = ozet(kayit, model_adi)
    sahte_veri.sonuc_yaz(os.path.join(KOK, "sonuclar", "y-3_hata.csv"), satirlar)
    algi = dict(model=model_adi,
                en_kucuk_nesne_px=o["en_kucuk_px"],
                en_kucuk_nesne_m=round(o["en_kucuk_m"], 2) if o["en_kucuk_m"] else None,
                onerilen_pay_hucre=o["pay"],
                not_=(f"UAVid test seti (eğik, ~50 m), araç + insan. en_kucuk_nesne_px: 4K resimde √alan; bundan küçüklerin "
                      f"yarısından azı bulunuyor (model resmi 1280'e küçültüyor, gözünde ~1/3'ü). Metre, aynı resimdeki "
                      f"araçlar 4.5 m sayılarak yaklaşık. Pay: bulunan nesnelerde kutu kenarı hatası %95'te "
                      f"{o['kenar_hatasi_p95_m']:.2f} m (medyan {o['kenar_hatasi_medyan_m']:.2f} m), hücre {HUCRE_M} m. "
                      f"Bu boydan küçük, hiç bulunmayan nesneye pay işe yaramaz: alçak uçuşta (5-10 m) insan/direk "
                      f"için ayrıca kural gerekir." if o["kenar_hatasi_p95_m"] else "yeterli eşleşme yok"))
    with open(os.path.join(args.veri, "algi_ozet.json"), "w", encoding="utf-8") as f:
        json.dump(algi, f, ensure_ascii=False, indent=1)
    print("\n".join(md))
    print("\nEnes'e (veri/algi_ozet.json):", json.dumps(algi, ensure_ascii=False, indent=1))
    print("CSV: sonuclar/y-3_hata.csv · örnekler: veri/_hata_ornekleri/")
    print("TAMAM")


if __name__ == "__main__":
    main()
