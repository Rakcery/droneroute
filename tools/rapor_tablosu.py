"""
sonuclar/ klasöründeki ölçüm CSV'lerini (S-6 biçimi) okuyup docs/olcumler.md içine tablo yazar.
Çalıştır:  python tools/rapor_tablosu.py

Her kart (Y-0, E-1, ...) için bir tablo çıkar: satırlar yöntem, sütunlar metrik.
Sahne "hepsi" diye bir özet varsa sadece onu gösterir (ayrıntı CSV'de duruyor).
Tablonun altındaki yorum cümlesi YORUMLAR sözlüğünden gelir; cümleyi kartın sahibi onaylar.
"""
import csv                                   # CSV okumak için
import glob                                  # klasördeki dosyaları bulmak için
import os                                    # dosya yolları için

KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # deponun ana klasörü
CIKTI = os.path.join(KOK, "docs", "olcumler.md")                       # yazılacak dosya

# Her kartın altına yazılacak tek cümle. Sahibi onaylamadan "onay bekliyor" kalır.
YORUMLAR = {
    "Y-0": "(onay bekliyor: Yağmur)",
    "E-1": "(onay bekliyor: Enes)",
}


def oku():
    """sonuclar/*.csv içindeki tüm satırları tek listede topla."""
    satirlar = []
    for yol in sorted(glob.glob(os.path.join(KOK, "sonuclar", "*.csv"))):   # her CSV dosyası
        with open(yol, encoding="utf-8") as f:
            for s in csv.DictReader(f):                                     # her satır bir sözlük
                s["dosya"] = os.path.basename(yol)                          # hangi dosyadan geldi
                satirlar.append(s)
    return satirlar


def tablo(kart_satirlari):
    """Bir kartın satırlarından Markdown tablo yaz: satır = yöntem, sütun = metrik."""
    if any(s["sahne"] == "hepsi" for s in kart_satirlari):                  # özet satırı varsa
        kart_satirlari = [s for s in kart_satirlari if s["sahne"] == "hepsi"]  # sadece özeti al
    yontemler, metrikler, deger = [], [], {}
    for s in kart_satirlari:
        if s["yontem"] not in yontemler:
            yontemler.append(s["yontem"])                                   # sırayı koru
        m = f'{s["metrik"]} ({s["birim"]})'                                 # sütun başlığı: metrik + birim
        if m not in metrikler:
            metrikler.append(m)
        deger[(s["yontem"], m)] = s["deger"]                                # tabloya girecek sayı
    sahneler = sorted({s["sahne"] for s in kart_satirlari})                 # hangi sahnelerden
    cizgi = ["| yöntem | " + " | ".join(metrikler) + " |",                  # başlık satırı
             "|---|" + "---|" * len(metrikler)]                              # ayraç satırı
    for y in yontemler:
        cizgi.append(f"| {y} | " + " | ".join(deger.get((y, m), "-") for m in metrikler) + " |")
    return cizgi, sahneler


def main():
    satirlar = oku()
    kartlar = []
    for s in satirlar:                                                      # kartları ilk görülme sırasıyla al
        if (s["kart"], s["deney"]) not in kartlar:
            kartlar.append((s["kart"], s["deney"]))
    out = ["# Ölçümler", "",
           "Bu dosya `python tools/rapor_tablosu.py` ile `sonuclar/*.csv` dosyalarından üretilir. Elle düzenleme,",
           "sayı değişecekse ölçümü tekrar koş ve bu betiği yeniden çalıştır.", ""]
    for kart, deney in kartlar:
        ks = [s for s in satirlar if s["kart"] == kart and s["deney"] == deney]
        cizgi, sahneler = tablo(ks)
        out += [f"## {kart} · {deney}", "",
                f"Kaynak: `sonuclar/{ks[0]['dosya']}` · sahne: {', '.join(sahneler)}", ""]
        out += cizgi
        out += ["", f"**Yorum:** {YORUMLAR.get(kart, '(onay bekliyor)')}", ""]
    os.makedirs(os.path.dirname(CIKTI), exist_ok=True)
    with open(CIKTI, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print(f"yazildi: docs/olcumler.md ({len(kartlar)} tablo)")


if __name__ == "__main__":
    main()
