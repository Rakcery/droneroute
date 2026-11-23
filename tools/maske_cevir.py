"""
Y-1 · Segmentasyon maskesini bizim YOLO etiketimize çevir (Sözleşme S-5).

Ne yapar: UAVid'in her etiket resminde her piksel bir renge boyanmış (bina kırmızımsı, ağaç yeşil ...).
Bu betik her sınıfın rengini ayrı ayrı bulur, aynı renkteki her bitişik lekeyi bir nesne sayar, lekenin
dış çizgisini (çokgen) çıkarır ve YOLO segmentasyon satırı olarak yazar:
    sinif x1 y1 x2 y2 x3 y3 ...     (0-1 arasına ölçeklenmiş)
Bizim sınıfımız olmayan renkler (yol, çimen, arka plan) atılır.

Girdi (ham, indirildiği gibi):   veri/_ham/uavid/uavid_train/seq1/Images/000000.png
                                  veri/_ham/uavid/uavid_train/seq1/Labels/000000.png
Çıktı (S-5):                      veri/uavid/seq1/images/000000.png
                                  veri/uavid/seq1/labels/000000.txt
                                  veri/uavid/seq1/meta.json
                                  veri/bolme.json   (uavid_train -> eğitim; uavid_val dizileri sırayla doğrulama / test)

VisDrone (kutu etiketli) da aynı betikte ama VARSAYILAN KAPALI (--visdrone ile açılır):
veri/_ham/visdrone/VisDrone2019-DET-val -> veri/visdrone/val_pilot (hep aynı 40 resim).
Neden kapalı: VisDrone sadece araç ve insanı etiketliyor; resimdeki bina ve ağaçlar etiketsiz. Eğitime
girerse modele "bina/ağaç = arka plan" öğretir, asıl düzeltmek istediğimiz zayıflığı büyütür.
İleride sadece araç/insan ölçümü için kullanılabilir.

Çalıştır (depo kökünden):
  python tools/maske_cevir.py                     # UAVid -> veri/uavid, önizlemeler veri/_onizleme/
  python tools/maske_cevir.py --visdrone          # + VisDrone (araç/insan ölçümü için, eğitime değil)
  python tools/maske_cevir.py --deneme            # sahte bir maskeyle kendi kendini dene
  python tools/veri_kontrol.py veri/              # sonra mutlaka kontrol et
Önizleme: veri/_onizleme/ altında her sahneden bir resim, etiketler üstüne çizili. Gözle bak: çizgiler
doğru nesnenin üstünde mi? (kırmızı bina, yeşil ağaç, pembe araç, sarı insan)

Bilinen sınır: yan yana park etmiş iki araba maskede tek leke olursa tek nesne çıkar. Elle kontrolde
(CVAT) bunlar ayrılır. Bina/ağaç için sorun değil: planlayıcı için önemli olan nerede engel olduğu.
"""
import argparse                              # komut satırı seçenekleri
import json                                  # meta.json ve bolme.json
import os                                    # klasörler
import shutil                                # görüntüyü kopyalamak için
import sys                                   # depo kökünü import yoluna eklemek için
import tempfile                              # --deneme için geçici klasör

import cv2                                   # resim okuma/yazma ve kontur bulma
import numpy as np                           # sayı tabloları

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # depo kökü
sys.path.insert(0, KOK)

# UAVid readme.txt'deki renkler (R, G, B) -> bizim sınıf numaramız (S-5: 0 vinc, 1 bina, 2 agac, 3 arac, 4 insan, 5 kus)
UAVID_RENK = {
    (128, 0, 0): 1,      # Building       -> bina
    (0, 128, 0): 2,      # Tree           -> agac
    (64, 0, 128): 3,     # Moving car     -> arac
    (192, 0, 192): 3,    # Static car     -> arac
    (64, 64, 0): 4,      # Human          -> insan
}                        # Road, Low vegetation, Background clutter: listede yok = atılır

UAVID_META = dict(kaynak="uavid", irtifa_m=50.0, egim_derece=None, hfov_derece=None,
                  lisans="CC BY-NC-SA 4.0 (Lyu et al., ISPRS 2020)")   # bilinmeyen değer null (README kuralı)

# VisDrone kategori numarası -> bizim sınıf. 0 "ignored region" ve 11 "others" atılır.
VISDRONE_SINIF = {1: 4, 2: 4,                                  # pedestrian, people -> insan
                  3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3}   # bisiklet, araba, van, kamyon, ... -> arac
VISDRONE_META = dict(kaynak="visdrone", irtifa_m=None, egim_derece=None, hfov_derece=None,
                     lisans="Araştırma amaçlı, ticari değil (VisDrone, Tianjin Univ.)")
VISDRONE_ADET = 40       # pilot sete giren VisDrone resmi (README: 40)

ONIZLEME_RENK = {0: (0, 165, 255), 1: (0, 0, 255), 2: (0, 200, 0), 3: (255, 0, 255), 4: (0, 255, 255), 5: (255, 255, 0)}  # BGR

# Sınıfa göre en küçük leke (resmin alanına oran). İnsan 4K'da ~20x40 piksel: tek eşik insanları da siliyordu.
MIN_ALAN_ORAN = {1: 0.0002, 2: 0.0002, 3: 0.00002, 4: 0.000005}   # bina, ağaç ~40x40 px; araç ~13x13; insan ~6x6 (4K'da)
SADELESTIR = 0.002       # çokgeni sadeleştirme: çevre uzunluğunun bu oranı kadar sapma...
EPS_MAX_PX = 3.0         # ...ama en fazla 3 piksel. Uzun ağaç şeritlerinde sınır kopmasın (seq5'teki zikzaklar)


def koprule(dis, delikler):
    """
    YOLO çokgeninde delik olamaz. Delikli bölgeyi (ör. ortasında ağaç olan bina bloğu) tek çokgene çevirmek
    için her deliği dış çizgiye ince bir "köprü" ile bağlarız: dış çizgide deliğe en yakın noktadan deliğe gir,
    deliğin etrafını dolaş, aynı noktadan geri çık. Çizildiğinde (fillPoly) delik boş kalır.
    """
    poli = [tuple(p) for p in dis]
    for d in delikler:
        d = [tuple(p) for p in d]
        P, D = np.array(poli, float), np.array(d, float)
        uz = ((P[:, None, :] - D[None, :, :]) ** 2).sum(-1)      # her dış nokta - her delik noktası uzaklığı
        i, j = np.unravel_index(np.argmin(uz), uz.shape)        # en yakın çift
        d = d[j:] + d[:j]                                       # deliği o noktadan başlat
        poli = poli[:i + 1] + d + [d[0]] + poli[i:]             # gir, dolaş, geri çık
    return np.array(poli, float)


def maske_satirlari(etiket_rgb, renk_sinif=UAVID_RENK):
    """
    Renkli etiket resmi (H x W x 3, RGB) -> YOLO segmentasyon satırları listesi.
    Her sınıf rengi için: o renkteki pikseller -> bitişik lekeler -> dış çizgi + delikler -> tek çokgen.
    """
    H, W = etiket_rgb.shape[:2]
    satirlar = []
    sinif_maske = {}                         # aynı sınıfa giden renkleri (hareketli + duran araba) birleştir
    for renk, sinif in renk_sinif.items():
        m = np.all(etiket_rgb == np.array(renk, np.uint8), axis=2)   # bu renkteki pikseller True
        sinif_maske[sinif] = sinif_maske.get(sinif, np.zeros((H, W), bool)) | m
    for sinif, m in sorted(sinif_maske.items()):
        min_alan = MIN_ALAN_ORAN.get(sinif, 0.0002) * H * W
        m = m.astype(np.uint8) * 255         # OpenCV 0/255 ister
        # RETR_CCOMP: dış çizgiler + onların içindeki delikler (hiyerarşi: [sonraki, önceki, ilk_çocuk, ebeveyn])
        konturlar, hiy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hiy is None:
            continue
        hiy = hiy[0]

        def sadelestir(k):
            eps = min(SADELESTIR * cv2.arcLength(k, True), EPS_MAX_PX)
            return cv2.approxPolyDP(k, eps, True).reshape(-1, 2)

        for i, k in enumerate(konturlar):
            if hiy[i][3] != -1 or cv2.contourArea(k) < min_alan:
                continue                     # delikleri ebeveynleriyle işle; küçük leke gürültü
            dis = sadelestir(k)
            if len(dis) < 3:
                continue
            delikler, c = [], hiy[i][2]      # ilk çocuk (delik), sonra kardeşleri
            while c != -1:
                if cv2.contourArea(konturlar[c]) >= min_alan:
                    d = sadelestir(konturlar[c])
                    if len(d) >= 3:
                        delikler.append(d)
                c = hiy[c][0]
            poli = koprule(dis, delikler) if delikler else dis.astype(float)
            noktalar = np.clip(poli / [W, H], 0, 1)             # piksel -> 0-1 arası
            satirlar.append(f"{sinif} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in noktalar))
    return satirlar


def geri_ciz_iou(etiket_rgb, satirlar, renk_sinif=UAVID_RENK):
    """
    Çeviriyi ölç: yazdığımız çokgenleri tekrar maskeye çiz, orijinal maskeyle karşılaştır.
    Sınıf başına IoU (kesişim / birleşim; 1.0 = birebir aynı) döner. Sınıf resimde yoksa yazılmaz.
    """
    H, W = etiket_rgb.shape[:2]
    out = {}
    for sinif in sorted(set(renk_sinif.values())):
        gercek = np.zeros((H, W), bool)
        for renk, s in renk_sinif.items():
            if s == sinif:
                gercek |= np.all(etiket_rgb == np.array(renk, np.uint8), axis=2)
        cizim = np.zeros((H, W), np.uint8)
        for satir in satirlar:
            p = satir.split()
            if int(p[0]) == sinif:
                pts = (np.array(p[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
                cv2.fillPoly(cizim, [pts], 1)
        if gercek.sum() > 0:
            out[sinif] = float((gercek & (cizim > 0)).sum() / (gercek | (cizim > 0)).sum())
    return out


def resim_kaydet(kaynak, hedef_klasor, ad):
    """Görüntüyü JPG olarak kaydeder (UAVid'in 15 MB'lık PNG'leri ~1.5 MB olur, YOLO için fark etmez)."""
    img = cv2.imread(kaynak, cv2.IMREAD_COLOR)
    if img is None:
        return None
    cv2.imwrite(os.path.join(hedef_klasor, ad + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img


def onizleme_yaz(img, satirlar, yol, genislik=1280):
    """
    Etiketleri resmin üstüne yarı saydam RENKLİ DOLGU + çizgi olarak çizip küçültülmüş bir JPG yazar.
    Dolgu sayesinde neyin etiketli olduğu tek bakışta görünür; boyanmamış yer = etiketsiz.
    """
    H, W = img.shape[:2]
    dolgu = img.copy()
    cizgi = []
    for s in satirlar:
        p = s.split()
        sinif, k = int(p[0]), [float(v) for v in p[1:]]
        renk = ONIZLEME_RENK[sinif]
        if len(k) == 4:                      # kutu: xc yc w h
            xc, yc, w, h = k
            pts = np.array([[xc - w / 2, yc - h / 2], [xc + w / 2, yc - h / 2],
                            [xc + w / 2, yc + h / 2], [xc - w / 2, yc + h / 2]])
        else:                                # çokgen
            pts = np.array(k).reshape(-1, 2)
        pts = (pts * [W, H]).round().astype(np.int32)
        cv2.fillPoly(dolgu, [pts], renk)
        cizgi.append((pts, renk))
    out = cv2.addWeighted(dolgu, 0.45, img, 0.55, 0)    # %45 renk, %55 fotoğraf
    for pts, renk in cizgi:
        cv2.polylines(out, [pts], True, renk, max(1, W // 1280))
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imwrite(yol, cv2.resize(out, (genislik, int(H * genislik / W))), [cv2.IMWRITE_JPEG_QUALITY, 85])


def dizi_cevir(ham_dizi, cikis_dizi, meta, onizleme=None, iou_listesi=None):
    """
    Bir UAVid dizisini (seqN: Images/ + Labels/) S-5 düzenine çevirir. Kaç resim çevrildiğini döner.
    iou_listesi verilirse her diziden ilk resmin geri-çizim IoU'sunu ekler (çeviri ne kadar doğru?).
    """
    img_in, lbl_in = os.path.join(ham_dizi, "Images"), os.path.join(ham_dizi, "Labels")
    if not os.path.isdir(lbl_in):
        return 0                             # etiketsiz dizi (test seti): atla
    img_out, lbl_out = os.path.join(cikis_dizi, "images"), os.path.join(cikis_dizi, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n = 0
    for dosya in sorted(os.listdir(lbl_in)):
        ad, uzanti = os.path.splitext(dosya)
        if uzanti.lower() != ".png":
            continue
        etiket = cv2.imread(os.path.join(lbl_in, dosya), cv2.IMREAD_COLOR)
        if etiket is None:
            print(f"  [!] okunamadı: {dosya}")
            continue
        etiket = cv2.cvtColor(etiket, cv2.COLOR_BGR2RGB)   # OpenCV BGR okur, readme'deki renkler RGB
        satirlar = maske_satirlari(etiket)
        img = resim_kaydet(os.path.join(img_in, dosya), img_out, ad)   # görüntü: PNG -> JPG
        if img is None:
            print(f"  [!] görüntü yok ya da okunamadı: {dosya}")
            continue
        with open(os.path.join(lbl_out, ad + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(satirlar) + "\n")
        if n == 0:                           # her diziden ilk resim: önizleme + doğruluk ölçümü
            if onizleme:
                onizleme_yaz(img, satirlar, onizleme)
            if iou_listesi is not None:
                iou_listesi.append(geri_ciz_iou(etiket, satirlar))
        n += 1
    with open(os.path.join(cikis_dizi, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return n


def bolme_oku(bolme_yolu, onek):
    """bolme.json'u okur; bu kaynağın (onek) eski satırlarını siler ki yeniden yazılabilsin."""
    bolme = {"egitim": [], "dogrulama": [], "test": []}
    if os.path.isfile(bolme_yolu):
        with open(bolme_yolu, encoding="utf-8") as f:
            eski = json.load(f)
        for b in bolme:
            bolme[b] = [s for s in eski.get(b, []) if not s.startswith(onek)]
    return bolme


def bolme_yaz(bolme_yolu, bolme):
    os.makedirs(os.path.dirname(bolme_yolu) or ".", exist_ok=True)
    with open(bolme_yolu, "w", encoding="utf-8") as f:
        json.dump(bolme, f, ensure_ascii=False, indent=1)


def uavid_kokunu_bul(ham):
    """Zip açılınca araya 'uavid_v1.5_official_release_image' klasörü girebiliyor: uavid_train'i nerede bulursa onu döner."""
    if os.path.isdir(os.path.join(ham, "uavid_train")):
        return ham
    for d in sorted(os.listdir(ham)) if os.path.isdir(ham) else []:
        if os.path.isdir(os.path.join(ham, d, "uavid_train")):
            return os.path.join(ham, d)
    return ham


def uavid_cevir(ham, cikis, bolme_yolu, onizleme_klasor=None):
    """veri/_ham/uavid -> veri/uavid. uavid_train dizileri eğitime, uavid_val dizileri sırayla doğrulama/test'e."""
    ham = uavid_kokunu_bul(ham)
    bolme = bolme_oku(bolme_yolu, "uavid/")   # başka kaynakların bölmesi korunur
    toplam = 0
    iou = []                                 # her diziden bir resmin sınıf başına IoU'su
    for alt, hedef in (("uavid_train", "egitim"), ("uavid_val", None)):
        kok = os.path.join(ham, alt)
        if not os.path.isdir(kok):
            print(f"[!] {kok} yok, atlandı")
            continue
        diziler = sorted((d for d in os.listdir(kok) if d.startswith("seq")), key=lambda s: int(s[3:]))
        for i, dizi in enumerate(diziler):
            oniz = os.path.join(onizleme_klasor, f"uavid_{dizi}.jpg") if onizleme_klasor else None
            n = dizi_cevir(os.path.join(kok, dizi), os.path.join(cikis, dizi), UAVID_META, oniz, iou)
            if n == 0:
                continue
            # val dizileri sırayla: biri doğrulama, biri test. Aynı dizi asla iki tarafa düşmez.
            yer = hedef or ("dogrulama" if i % 2 == 0 else "test")
            bolme[yer].append(f"uavid/{dizi}")
            toplam += n
            print(f"  {alt}/{dizi}: {n} resim -> {yer}")
    bolme_yaz(bolme_yolu, bolme)
    if iou:                                  # çeviri doğruluğu: çokgenleri geri çizip orijinal maskeyle karşılaştır
        adlar = {1: "bina", 2: "agac", 3: "arac", 4: "insan"}
        print("Çeviri doğruluğu (geri çizim IoU, 1.00 = birebir; her diziden 1 resim):")
        for s, ad in adlar.items():
            v = [d[s] for d in iou if s in d]
            if v:
                print(f"  {ad:6s} ortalama {np.mean(v):.3f}  en kötü {min(v):.3f}")
    print(f"UAVid: {toplam} resim çevrildi; bölme: " + ", ".join(f"{b} {len(v)} dizi" for b, v in bolme.items()))
    return toplam


def visdrone_satirlari(ann_yolu, W, H):
    """VisDrone etiketi (x,y,w,h,skor,kategori,kesik,örtülü; piksel) -> YOLO kutu satırları."""
    satirlar = []
    with open(ann_yolu, encoding="utf-8") as f:
        for s in f:
            p = [v for v in s.strip().split(",") if v != ""]
            if len(p) < 6:
                continue
            x, y, w, h, skor, kat = (int(float(v)) for v in p[:6])
            if skor == 0 or kat not in VISDRONE_SINIF or w <= 0 or h <= 0:
                continue                     # skor 0 = "burayı yok say" bölgesi; 0/11 = bizim sınıf değil
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)   # resmin dışına taşmasın
            if x2 <= x1 or y2 <= y1:
                continue
            satirlar.append(f"{VISDRONE_SINIF[kat]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return satirlar


def visdrone_cevir(ham, cikis, bolme_yolu, adet=VISDRONE_ADET, onizleme_klasor=None, tohum=0):
    """VisDrone2019-DET-val'den 'adet' resim seçer -> veri/visdrone/val_pilot (hepsi eğitime)."""
    kok = os.path.join(ham, "VisDrone2019-DET-val")
    if not os.path.isdir(kok):
        print(f"[!] {kok} yok, VisDrone atlandı")
        return 0
    adlar = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(kok, "images")) if f.lower().endswith(".jpg"))
    rng = np.random.default_rng(tohum)       # hep aynı tohum = hep aynı 40 resim (tekrar üretilebilir)
    secilen = sorted(rng.choice(adlar, size=min(adet, len(adlar)), replace=False))
    sahne = os.path.join(cikis, "val_pilot")
    img_out, lbl_out = os.path.join(sahne, "images"), os.path.join(sahne, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for i, ad in enumerate(secilen):
        img = cv2.imread(os.path.join(kok, "images", ad + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        satirlar = visdrone_satirlari(os.path.join(kok, "annotations", ad + ".txt"), W, H)
        shutil.copy2(os.path.join(kok, "images", ad + ".jpg"), os.path.join(img_out, ad + ".jpg"))
        with open(os.path.join(lbl_out, ad + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(satirlar) + "\n")
        if onizleme_klasor and i < 3:        # ilk 3 resmin önizlemesi
            onizleme_yaz(img, satirlar, os.path.join(onizleme_klasor, f"visdrone_{ad}.jpg"))
    with open(os.path.join(sahne, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(VISDRONE_META, f, ensure_ascii=False, indent=1)
    bolme = bolme_oku(bolme_yolu, "visdrone/")
    bolme["egitim"].append("visdrone/val_pilot")
    bolme_yaz(bolme_yolu, bolme)
    print(f"VisDrone: {len(secilen)} resim çevrildi -> egitim")
    return len(secilen)


def deneme():
    """Sahte bir UAVid dizisi üret (bina dikdörtgeni, ağaç dairesi, araba, yol), çevir, sonucu kontrol et."""
    from tools.veri_kontrol import kontrol
    with tempfile.TemporaryDirectory() as tmp:
        for alt, dizi in (("uavid_train", "seq1"), ("uavid_val", "seq16"), ("uavid_val", "seq17")):
            d = os.path.join(tmp, "_ham", "uavid", "uavid_v1.5_official_release_image", alt, dizi)   # zip'teki ara klasörle
            os.makedirs(os.path.join(d, "Images"))
            os.makedirs(os.path.join(d, "Labels"))
            etiket = np.zeros((400, 600, 3), np.uint8)                       # arka plan (0,0,0)
            etiket[:, 250:300] = (128, 64, 128)                              # yol: atılmalı
            etiket[50:200, 50:200] = (128, 0, 0)                             # bina
            cv2.circle(etiket, (450, 120), 60, (0, 128, 0), -1)               # ağaç
            etiket[300:340, 400:470] = (192, 0, 192)                         # park etmiş araba
            etiket[300:302, 10:12] = (64, 64, 0)                             # 4 piksellik insan: gürültü, atılmalı
            cv2.imwrite(os.path.join(d, "Labels", "000000.png"), cv2.cvtColor(etiket, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(d, "Images", "000000.png"), np.full((400, 600, 3), 120, np.uint8))
        veri = os.path.join(tmp, "veri")
        uavid_cevir(os.path.join(tmp, "_ham", "uavid"), os.path.join(veri, "uavid"), os.path.join(veri, "bolme.json"))
        # Sahte VisDrone: 1 resim; araba (4), yaya (1), "yok say" bölgesi (0) ve resimden taşan kutu
        vd = os.path.join(tmp, "_ham", "visdrone", "VisDrone2019-DET-val")
        os.makedirs(os.path.join(vd, "images"))
        os.makedirs(os.path.join(vd, "annotations"))
        cv2.imwrite(os.path.join(vd, "images", "0000001_00000_d_0000001.jpg"), np.full((100, 200, 3), 90, np.uint8))
        with open(os.path.join(vd, "annotations", "0000001_00000_d_0000001.txt"), "w") as f:
            f.write("10,10,40,20,1,4,0,0\n60,30,5,12,1,1,0,0\n0,0,50,50,0,0,0,0\n180,90,40,40,1,6,0,0\n")
        visdrone_cevir(os.path.join(tmp, "_ham", "visdrone"), os.path.join(veri, "visdrone"), os.path.join(veri, "bolme.json"))
        with open(os.path.join(veri, "visdrone", "val_pilot", "labels", "0000001_00000_d_0000001.txt")) as f:
            vs = [s.split() for s in f.read().split("\n") if s.strip()]
        assert [s[0] for s in vs] == ["3", "4", "3"], vs          # araba, insan, taşan kamyon (kırpılmış); "yok say" atıldı
        assert all(0 <= float(v) <= 1 for s in vs for v in s[1:]), vs
        with open(os.path.join(veri, "uavid", "seq1", "labels", "000000.txt")) as f:
            siniflar = sorted(int(s.split()[0]) for s in f.read().split("\n") if s.strip())
        assert siniflar == [1, 2, 3], f"beklenen bina, ağaç, araç; çıkan {siniflar}"
        with open(os.path.join(veri, "bolme.json")) as f:
            b = json.load(f)
        assert b["egitim"] == ["uavid/seq1", "visdrone/val_pilot"] and b["dogrulama"] == ["uavid/seq16"] and b["test"] == ["uavid/seq17"], b
        hatalar = kontrol(veri)
        assert hatalar == [], hatalar
        print("deneme: UAVid bina + ağaç + araç bulundu, yol ve gürültü atıldı; VisDrone araç/insan doğru, "
              "yok-say bölgesi atıldı, taşan kutu kırpıldı; bölme doğru, veri_kontrol temiz")


def main():
    ap = argparse.ArgumentParser(description="Y-1: UAVid maskeleri + VisDrone kutularını S-5 YOLO etiketine çevir")
    ap.add_argument("--veri", default=os.path.join(KOK, "veri"), help="veri kök klasörü (_ham/ burada)")
    ap.add_argument("--visdrone", action="store_true", help="VisDrone'u da çevir (bina/ağaç etiketsiz: eğitime koyma)")
    ap.add_argument("--deneme", action="store_true", help="sahte maskeyle kendi kendini dene")
    args = ap.parse_args()
    if args.deneme:
        deneme()
    else:
        v = args.veri
        bolme, oniz = os.path.join(v, "bolme.json"), os.path.join(v, "_onizleme")
        uavid_cevir(os.path.join(v, "_ham", "uavid"), os.path.join(v, "uavid"), bolme, oniz)
        if args.visdrone:
            visdrone_cevir(os.path.join(v, "_ham", "visdrone"), os.path.join(v, "visdrone"), bolme, onizleme_klasor=oniz)
        print(f"Önizlemeler: {oniz}")
    print("TAMAM")


if __name__ == "__main__":
    main()
