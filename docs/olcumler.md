# Ölçümler

Bu dosya `python tools/rapor_tablosu.py` ile `sonuclar/*.csv` dosyalarından üretilir. Elle düzenleme,
sayı değişecekse ölçümü tekrar koş ve bu betiği yeniden çalıştır.

## E-1 · kisaltma

Kaynak: `sonuclar/e-1_kisaltma.csv` · sahne: 300_rastgele

| yöntem | nokta_sayisi (adet) | uzunluk (hucre) |
|---|---|---|
| astar | 39.53 | 40.931 |
| astar+shortcut | 6.94 | 39.272 |

**Yorum:** (onay bekliyor: Enes)

## Y-0 · algilama

Kaynak: `sonuclar/y-0_algilama.csv` · sahne: hepsi

| yöntem | kesinlik (oran) | duyarlilik (oran) | sure (ms/kare) |
|---|---|---|---|
| renk | 0.9832 | 0.9608 | 9.5715 |
| kenar | 0.9338 | 0.9681 | 2.2954 |
| yolo_oracle | 0.9851 | 0.986 | 1.6678 |
| hibrit_oracle | 0.9773 | 0.987 | 11.2601 |

**Yorum:** (onay bekliyor: Yağmur)
