# Ölçümler

Bu dosya `python tools/rapor_tablosu.py` ile `sonuclar/*.csv` dosyalarından üretilir. Elle düzenleme,
sayı değişecekse ölçümü tekrar koş ve bu betiği yeniden çalıştır.

## E-1 · kisaltma

Kaynak: `sonuclar/e-1_kisaltma.csv` · sahne: 300_rastgele

| yöntem | nokta_sayisi (adet) | uzunluk (hucre) |
|---|---|---|
| astar | 39.53 | 40.931 |
| astar+shortcut | 6.94 | 39.272 |

**Yorum:** 300 rastgele haritada görüş hattı kısaltması, drone'a verilen rota noktası sayısını ortalama 39,5'ten 6,9'a indirdi (%82 az), rota uzunluğu da %4 kısaldı (40,9 → 39,3 hücre). (onay: Enes)

## Y-0 · algilama

Kaynak: `sonuclar/y-0_algilama.csv` · sahne: hepsi

| yöntem | kesinlik (oran) | duyarlilik (oran) | sure (ms/kare) |
|---|---|---|---|
| renk | 0.9832 | 0.9608 | 9.5715 |
| kenar | 0.9338 | 0.9681 | 2.2954 |
| yolo_oracle | 0.9851 | 0.986 | 1.6678 |
| hibrit_oracle | 0.9773 | 0.987 | 11.2601 |

**Yorum:** 6 sahnede klasik yöntemlerden renk algılama daha az yanlış alarm veriyor (kesinlik 0,98) ama biraz daha fazla engel kaçırıyor (duyarlılık 0,96, kenarda 0,97). Kenar algılama ise yaklaşık 4 kat daha hızlı (2,3 ms/kare). yolo_oracle ve hibrit_oracle satırları gerçek YOLO modeli değil, simülasyondaki kusursuz dedektörle ölçüldü. Gerçek model Y-2'de ölçülecek. (onay: Yağmur)
