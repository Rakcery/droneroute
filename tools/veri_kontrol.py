"""
Y-1 · Etiketli veri klasörünü Sözleşme S-5'e göre kontrol et.

Ne yapar: veri/ altındaki her sahne klasörüne bakar ve şu soruları sorar:
  1. Her görüntünün bir etiket dosyası var mı? (ve her etiketin bir görüntüsü)
  2. Etiket satırları doğru biçimde mi?
       kutu:          sinif xc yc w h            (5 sayı)
       segmentasyon:  sinif x1 y1 x2 y2 x3 y3 ...  (sınıf + en az 3 nokta, çift sayıda koordinat)
  3. Bütün koordinatlar 0-1 arasında mı? Kutunun genişliği/yüksekliği 0'dan büyük mü?
  4. Sınıf numarası 0-5 arasında mı? (0 vinc, 1 bina, 2 agac, 3 arac, 4 insan, 5 kus)
  5. meta.json var mı, içinde istenen 5 alan var mı?
  6. (bolme.json varsa) Her sahne eğitim / doğrulama / test'ten TAM OLARAK birinde mi?
     Aynı sahne iki tarafta olursa test "kopya çekmiş" olur.

Hata bulursa hepsini listeler ve HATA ile biter; temizse sınıf sayımını basar ve TAMAM ile biter.
Boş etiket dosyası geçerlidir: o görüntüde hiç nesne yok demek.

Çalıştır (depo kökünden):
  python tools/veri_kontrol.py veri/                # gerçek veri klasörü
  python tools/veri_kontrol.py --deneme             # sahte veriyle kendi kendini dene (veri gerekmez)
"""
import argparse                              # komut satırı seçenekleri
import json                                  # meta.json ve bolme.json okumak için
import os                                    # klasör gezmek için
import sys                                   # depo kökünü import yoluna eklemek için
import tempfile                              # --deneme için geçici klasör

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # depo kökü
sys.path.insert(0, KOK)

SINIFLAR = ["vinc", "bina", "agac", "arac", "insan", "kus"]          # S-5 sırası (sahte_veri.SINIFLAR ile aynı)
META_ALANLARI = {"kaynak", "irtifa_m", "egim_derece", "hfov_derece", "lisans"}   # S-5 meta.json alanları
GORUNTU_UZANTILARI = (".jpg", ".jpeg", ".png")
BOLMELER = ("egitim", "dogrulama", "test")


def satir_kontrol(satir):
    """Tek bir etiket satırını kontrol eder. Sorun yoksa (sinif, None), varsa (None, 'hata açıklaması')."""
    parcalar = satir.split()                 # boşluklardan böl
    try:
        sayilar = [float(p) for p in parcalar]   # hepsi sayı olmalı
    except ValueError:
        return None, "sayı olmayan değer var"
    if not sayilar or sayilar[0] != int(sayilar[0]):
        return None, "sınıf tam sayı değil"
    sinif = int(sayilar[0])
    if not 0 <= sinif < len(SINIFLAR):
        return None, f"sınıf {sinif} geçersiz (0-{len(SINIFLAR) - 1} olmalı)"
    koord = sayilar[1:]                      # sınıftan sonraki her şey koordinat
    if len(koord) == 4:                      # kutu: xc yc w h
        if koord[2] <= 0 or koord[3] <= 0:
            return None, "kutunun genişliği ya da yüksekliği 0"
    elif len(koord) < 6 or len(koord) % 2:   # segmentasyon: en az 3 nokta (x, y çiftleri)
        return None, f"{len(koord)} koordinat: kutu için 4, poligon için çift sayıda ve en az 6 olmalı"
    if any(not 0 <= v <= 1 for v in koord):
        return None, "koordinat 0-1 arasında değil (YOLO biçimi görüntü boyuna bölünmüş ister)"
    return sinif, None


def sahne_kontrol(sahne_yolu, hatalar, sayim):
    """Bir sahne klasörünü (images/ + labels/ + meta.json) kontrol eder. Bulduğu hataları listeye ekler."""
    ad = os.path.relpath(sahne_yolu)         # mesajlarda kısa yol göster
    img_dir = os.path.join(sahne_yolu, "images")
    lbl_dir = os.path.join(sahne_yolu, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        hatalar.append(f"{ad}: images/ ve labels/ klasörlerinin ikisi de olmalı")
        return 0
    # Dosya adlarını uzantısız karşılaştır: foto_001.jpg <-> foto_001.txt
    goruntuler = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(GORUNTU_UZANTILARI)}
    etiketler = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")}
    for eksik in sorted(goruntuler - etiketler):
        hatalar.append(f"{ad}: {eksik} görüntüsünün etiketi yok")
    for fazla in sorted(etiketler - goruntuler):
        hatalar.append(f"{ad}: {fazla}.txt etiketinin görüntüsü yok")
    for isim in sorted(etiketler):
        with open(os.path.join(lbl_dir, isim + ".txt"), encoding="utf-8") as f:
            for no, satir in enumerate(f, 1):    # satır numarası 1'den başlasın
                if not satir.strip():
                    continue                 # boş satır sorun değil
                sinif, hata = satir_kontrol(satir)
                if hata:
                    hatalar.append(f"{ad}/labels/{isim}.txt satır {no}: {hata}")
                else:
                    sayim[sinif] += 1        # sınıf sayımı (dengesizliği görmek için)
    meta_yolu = os.path.join(sahne_yolu, "meta.json")
    if not os.path.isfile(meta_yolu):
        hatalar.append(f"{ad}: meta.json yok")
    else:
        try:
            with open(meta_yolu, encoding="utf-8") as f:
                eksik = META_ALANLARI - set(json.load(f))
            if eksik:
                hatalar.append(f"{ad}: meta.json'da eksik alan: {', '.join(sorted(eksik))}")
        except json.JSONDecodeError:
            hatalar.append(f"{ad}: meta.json okunamadı (JSON biçimi bozuk)")
    return len(goruntuler)


def sahneleri_bul(kok):
    """S-5 klasör düzeni: <kok>/<kaynak>/<sahne>/. 'kaynak/sahne' adlarını döner (ör. 'uavid/seq1')."""
    out = []
    for kaynak in sorted(os.listdir(kok)):
        kaynak_yolu = os.path.join(kok, kaynak)
        if not os.path.isdir(kaynak_yolu) or kaynak.startswith("_"):
            continue                         # README.md, bolme.json gibi dosyaları ve _ham/ (indirilen ham veri) klasörünü atla
        for sahne in sorted(os.listdir(kaynak_yolu)):
            if os.path.isdir(os.path.join(kaynak_yolu, sahne)):
                out.append(f"{kaynak}/{sahne}")
    return out


def bolme_kontrol(kok, sahneler, hatalar):
    """bolme.json: {"egitim": [...], "dogrulama": [...], "test": [...]} sahne adları. Her sahne tam bir yerde olmalı."""
    yol = os.path.join(kok, "bolme.json")
    if not os.path.isfile(yol):
        print("[i] bolme.json yok, bölme kontrolü atlandı")
        return
    with open(yol, encoding="utf-8") as f:
        bolme = json.load(f)
    nerede = {}                              # sahne -> hangi bölmelerde geçiyor
    for b in BOLMELER:
        for s in bolme.get(b, []):
            nerede.setdefault(s, []).append(b)
    for s, yerler in nerede.items():
        if len(yerler) > 1:
            hatalar.append(f"bolme.json: {s} birden fazla yerde ({', '.join(yerler)}): test kopya çeker")
        if s not in sahneler:
            hatalar.append(f"bolme.json: {s} diye bir sahne klasörü yok")
    for s in sahneler:
        if s not in nerede:
            hatalar.append(f"bolme.json: {s} hiçbir bölmede değil")


def kontrol(kok):
    """Bütün klasörü kontrol eder. Hata listesini döner (boşsa her şey yolunda)."""
    hatalar = []
    sayim = [0] * len(SINIFLAR)
    sahneler = sahneleri_bul(kok)
    if not sahneler:
        return [f"{kok}: hiç sahne klasörü yok (beklenen: {kok}/<kaynak>/<sahne>/images ...)"]
    toplam = sum(sahne_kontrol(os.path.join(kok, s), hatalar, sayim) for s in sahneler)
    bolme_kontrol(kok, sahneler, hatalar)
    print(f"{len(sahneler)} sahne, {toplam} görüntü, {sum(sayim)} nesne")
    for ad, n in zip(SINIFLAR, sayim):
        print(f"  {ad:6s} {n}")              # hangi sınıftan az var, buradan görünür
    return hatalar


def deneme():
    """Sahte veriyle kendi kendini dene: önce temiz klasör TAMAM vermeli, sonra bozduğumuz her şeyi yakalamalı."""
    import sahte_veri
    with tempfile.TemporaryDirectory() as kok:
        for sahne in ("sahne1", "sahne2"):
            sahte_veri.sahte_etiket_klasoru(os.path.join(kok, "sahte", sahne), n=5, seed=len(sahne))
        with open(os.path.join(kok, "bolme.json"), "w") as f:
            json.dump({"egitim": ["sahte/sahne1"], "dogrulama": [], "test": ["sahte/sahne2"]}, f)
        assert kontrol(kok) == [], "temiz sahte veri hata vermemeli"

        # Şimdi bilerek boz: her bozukluk yakalanmalı
        lbl = os.path.join(kok, "sahte", "sahne1", "labels")
        with open(os.path.join(lbl, "sahte_0000.txt"), "a") as f:
            f.write("7 0.5 0.5 0.1 0.1\n")   # geçersiz sınıf
            f.write("1 1.5 0.5 0.1 0.1\n")   # 0-1 dışında koordinat
            f.write("2 0.5 0.5 0.1\n")       # eksik sayı
        os.remove(os.path.join(lbl, "sahte_0001.txt"))   # etiketi olmayan görüntü
        with open(os.path.join(kok, "bolme.json"), "w") as f:
            json.dump({"egitim": ["sahte/sahne1", "sahte/sahne2"], "test": ["sahte/sahne2"]}, f)   # sahne2 iki yerde
        hatalar = kontrol(kok)
        beklenen = ["sınıf 7", "0-1 arasında değil", "3 koordinat", "etiketi yok", "birden fazla yerde"]
        for b in beklenen:
            assert any(b in h for h in hatalar), f"yakalanmadı: {b}"
        print(f"bozulmuş klasörde {len(hatalar)} hata yakalandı, beklenen {len(beklenen)} tür hatanın hepsi var")


def main():
    ap = argparse.ArgumentParser(description="Y-1: S-5 etiketli veri kontrolü")
    ap.add_argument("klasor", nargs="?", default=os.path.join(KOK, "veri"), help="veri kök klasörü")
    ap.add_argument("--deneme", action="store_true", help="sahte veriyle kendi kendini dene")
    args = ap.parse_args()
    if args.deneme:
        deneme()
        print("TAMAM")
        return
    hatalar = kontrol(args.klasor)
    for h in hatalar:
        print("  [!]", h)
    if hatalar:
        print(f"HATA: {len(hatalar)} sorun")
        sys.exit(1)
    print("TAMAM")


if __name__ == "__main__":
    main()
