"""
Y-0 · Algılama yöntemlerini karşılaştır (renk, kenar, yolo, hibrit).

Ne yapar: Her yöntemi AYNI 6 simülasyon sahnesinde çalıştırır. Her karede yöntemin "dolu" dediği hücreleri
sahnenin cevap anahtarıyla (gerçek engel maskesi) karşılaştırır ve üç sayı çıkarır:
  kesinlik   = TP / (TP + FP)   "dolu" dediklerimizin ne kadarı gerçekten dolu? (düşükse: yanlış alarm çok)
  duyarlılık = TP / (TP + FN)   gerçek engellerin ne kadarını bulduk?        (düşükse: KAÇIRIYORUZ, tehlikeli)
  ms/kare    = algılamanın bir karede kaç milisaniye sürdüğü (sahne çizimi hariç)

Ölçüm gerçek sistemdeki gibi ZAMAN FİLTRESİNDEN GEÇMİŞ hücrelerle yapılır (OccupancyFilter), çünkü
planlayıcıya giden şey o. Güvenlik payı ölçüme girmez (pay planlamanın işi, algının değil).

YOLO ve hibrit için gerçek model indirilmez: SimOracle ("kusursuz YOLO") kullanılır. Yani bu iki satır
"YOLO hiç yanılmasaydı boru hattının geri kalanı (kutu → maske → hücre → filtre) ne kadar doğru olurdu?"
sorusunu cevaplar. Gerçek modelin sayıları Y-2/Y-3'te gelecek.

Çalıştır (depo kökünden):
  python tools/olcum_algilama.py              # 6 sahne x 160 kare, ~1-2 dk
  python tools/olcum_algilama.py --kare 30    # hızlı deneme
Çıktı: ekrana tablo (README'ye yapıştırılabilir Markdown) + sonuclar/y-0_algilama.csv (Sözleşme S-6).
"""
import argparse                              # komut satırı seçenekleri (--kare, --sahne ...)
import os                                    # dosya yolları
import sys                                   # depo kökünü import yoluna eklemek için
import time                                  # süre ölçmek için

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ klasörünün bir üstü = depo kökü
sys.path.insert(0, KOK)                      # böylece drone_vision, sahte_veri vb. import edilebilir

import numpy as np                           # sayı tabloları

import drone_vision as V                     # SimScene, ObstacleDetector, OccupancyFilter
import drone_yolo as Y                       # YoloDetector, SimOracle ("kusursuz YOLO")
import sahte_veri                            # sonuc_yaz: S-6 biçiminde CSV yazar

YONTEMLER = ("renk", "kenar", "yolo", "hibrit")   # karşılaştırılan dört yöntem (ObstacleDetector.METHODS)
# Rapordaki adlar: yolo ve hibrit burada gerçek modelle değil SimOracle ile ölçülüyor, adından belli olsun.
# Gerçek model Y-2'de ölçülünce o satırlar düz "yolo" / "hibrit" adıyla gelecek.
ETIKET = {"renk": "renk", "kenar": "kenar", "yolo": "yolo_oracle", "hibrit": "hibrit_oracle"}


def dedektor_kur(yontem, sahne):
    """Bir yöntem için algılayıcıyı hazırlar. Her sahne ve her yöntem için sıfırdan kurulur (adil olsun)."""
    det = V.ObstacleDetector(yontem)                         # yöntemi seç
    det.calibrate(sahne.background())                        # renk yöntemi boş zemini öğrensin (hibrit de kullanır)
    if yontem in ("yolo", "hibrit"):                         # YOLO isteyen yöntemler
        det.yolo = Y.YoloDetector(Y.SimOracle(sahne))        # gerçek model yerine sahnenin kendisi (kusursuz)
    return det


def sahne_olc(yontem, tohum, kare_sayisi):
    """
    Tek yöntem, tek sahne. Dönen: (TP, FP, FN, toplam_ms, kare).
    TP/FP/FN bütün karelerin toplamı; oranlar sonra hesaplanır (kare kare ortalamaktan daha sağlam).
    """
    sahne = V.SimScene(tohum)                # aynı tohum = her yöntem için birebir aynı sahne ve hareketler
    det = dedektor_kur(yontem, sahne)
    filtre = V.OccupancyFilter()             # zaman filtresi: 2 kare dolu → dolu, 4 kare boş → boş
    tp = fp = fn = 0                         # sayaçlar
    toplam_ms = 0.0                          # algılamanın toplam süresi
    for _ in range(kare_sayisi):
        kare, gercek_maske = sahne.render()  # kamera karesi + cevap anahtarı (piksel bazında)
        sahne.step()                         # sahneyi bir kare ilerlet (kamyon, vinç, kuş hareket eder)

        t0 = time.perf_counter()             # --- süre ölçümü başlar: sadece algılama ---
        maske = det.mask(kare)               # kare → engel maskesi (piksel)
        hucreler = V.ObstacleDetector.to_cells(maske)   # maske → 28x40 dolu/boş tablo
        tahmin = filtre.update(hucreler)     # zaman filtresi: planlayıcıya giden tablo
        toplam_ms += (time.perf_counter() - t0) * 1000  # --- süre ölçümü biter ---

        gercek = V.ObstacleDetector.to_cells(gercek_maske)   # cevap anahtarını da aynı şekilde hücreye çevir
        tp += int((gercek & tahmin).sum())   # dolu ve "dolu" dedik: doğru bulma
        fp += int((~gercek & tahmin).sum())  # boş ama "dolu" dedik: yanlış alarm
        fn += int((gercek & ~tahmin).sum())  # dolu ama "boş" dedik: KAÇIRMA
    return tp, fp, fn, toplam_ms, kare_sayisi


def oranlar(tp, fp, fn):
    """Kesinlik ve duyarlılık. max(..., 1): hiç tahmin yoksa sıfıra bölme olmasın."""
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    ap = argparse.ArgumentParser(description="Y-0: algılama yöntemlerini ölç")
    ap.add_argument("--sahne", type=int, default=6, help="kaç sahne (tohum 0..n-1), varsayılan 6")
    ap.add_argument("--kare", type=int, default=160, help="sahne başına kare, varsayılan 160 (test_vision.py ile aynı)")
    ap.add_argument("--cikti", default=os.path.join(KOK, "sonuclar", "y-0_algilama.csv"), help="CSV yolu")
    args = ap.parse_args()

    satirlar = []                            # CSV satırları (S-6)
    ozet = {}                                # yöntem -> (kesinlik, duyarlılık, ms) ekrana basmak için

    def ekle(yontem, sahne, metrik, deger, birim):
        """S-6 başlığına uyan tek bir CSV satırı ekler."""
        satirlar.append(dict(kart="Y-0", deney="algilama", yontem=ETIKET[yontem], sahne=sahne,
                             metrik=metrik, deger=round(deger, 4), birim=birim))

    for yontem in YONTEMLER:
        T = [0, 0, 0, 0.0, 0]                # bütün sahnelerin toplamı: tp, fp, fn, ms, kare
        for tohum in range(args.sahne):
            tp, fp, fn, ms, n = sahne_olc(yontem, tohum, args.kare)
            k, d = oranlar(tp, fp, fn)
            ekle(yontem, str(tohum), "kesinlik", k, "oran")      # sahne bazında satırlar
            ekle(yontem, str(tohum), "duyarlilik", d, "oran")
            ekle(yontem, str(tohum), "sure", ms / n, "ms/kare")
            for i, v in enumerate((tp, fp, fn, ms, n)):
                T[i] += v                    # toplama ekle
            print(f"  {ETIKET[yontem]:13s} sahne {tohum}: kesinlik={k:.3f} duyarlilik={d:.3f} {ms / n:.1f} ms/kare")
        k, d = oranlar(T[0], T[1], T[2])     # bütün sahneler birlikte
        ms = T[3] / T[4]
        ekle(yontem, "hepsi", "kesinlik", k, "oran")
        ekle(yontem, "hepsi", "duyarlilik", d, "oran")
        ekle(yontem, "hepsi", "sure", ms, "ms/kare")
        ozet[yontem] = (k, d, ms)

    sahte_veri.sonuc_yaz(args.cikti, satirlar)   # S-6 CSV

    # README'ye yapıştırılabilir tablo
    print(f"\n{args.sahne} sahne x {args.kare} kare, hücre bazında, zaman filtresinden sonra, güvenlik payı hariç")
    print("YOLO ve hibrit: SimOracle (kusursuz YOLO), gerçek model değil\n")
    print("| Yöntem | Kesinlik | Duyarlılık | ms/kare |")
    print("|---|---|---|---|")
    for yontem, (k, d, ms) in ozet.items():
        print(f"| {ETIKET[yontem]} | {k:.3f} | {d:.3f} | {ms:.1f} |")
    print(f"\nCSV: {os.path.relpath(args.cikti, KOK)}")
    print("TAMAM")


if __name__ == "__main__":
    main()
