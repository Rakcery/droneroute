# DroneRoute veri seti planı (Y-1)

YOLO'yu kendi sınıflarımızla (özellikle **bina** ve **ağaç**, yukarıdan) eğitmek için toplanacak verinin planı.
Biçim: Arayüz Sözleşmesi **S-5** (`docs/SOZLESME.md`). Kontrol: `python tools/veri_kontrol.py veri/`.

> **Görüntüler depoya girmez.** Bu klasörde depoda sadece bu README, `bolme.json` ve kontrol aracı durur.
> Görüntüler ve etiketler Drive'da (bağlantı: _Lider'den alınacak_). `git status`'ta `.jpg`/`.png` görürsen ekleme.

## Sınıflar (S-5)

| No | Sınıf | Ne sayılır |
|---|---|---|
| 0 | vinc | Kule vinç (kule + kol), mobil vinç |
| 1 | bina | Ev, apartman, şantiye binası, inşaat hâlindeki bina, çatı |
| 2 | agac | Gövdesi ve tacı olan ağaç (çalı, çimen değil) |
| 3 | arac | Otomobil, kamyon, otobüs, minibüs, iş makinesi, motosiklet, bisiklet |
| 4 | insan | Ayakta, yürüyen ya da oturan insan |
| 5 | kus | Kuş, başka bir drone |

## Kaynaklar ve lisanslar

| Kaynak | Ne var | Bakış | Bizim sınıflar | Lisans | Durum |
|---|---|---|---|---|---|
| **UAVid** (uavid.nl) | 30 video dizisi, 300 etiketli 4K kare, piksel bazında segmentasyon | Eğik, ~50 m | bina, agac, arac, insan | CC BY-NC-SA 4.0 (ticari değil, atıf, aynı lisansla paylaş) | Asıl kaynak: bina/ağaç için en uygun |
| **Semantic Drone Dataset** (TU Graz) | 400 eğitim görüntüsü, 20 sınıflı segmentasyon | Tepeden (nadir), 5-30 m | bina (roof), agac, arac, insan | Sadece ticari olmayan kullanım (araştırma, eğitim); atıf zorunlu; indirmek için form | Form doldurulacak |
| **VisDrone2019-DET** | 8 629 görüntü, 10 sınıflı kutu | Çeşitli, tepeden/eğik | arac, insan | Resmi depoda açık lisans yok, araştırma amaçlı yayımlanmış; dağıtımlarda CC BY-NC-SA 3.0 olarak geçiyor | Bina/ağaç yok; sadece araç/insan dengesi için |
| **Kampüs çekimleri** | Kendi videolarımız | Eğik | hepsi | Bizim (`ic-kullanim`); yüz/plaka varsa bulanıklaştır | Y-5'te |
| **Gazebo** (Kerem, Hafta 8+) | Otomatik etiketli sim görüntüsü | Eğik kamera | hepsi | Bizim | Y-4'te |

Üç açık veri setinin de lisansı **ticari olmayan** kullanıma izin veriyor. Kulüp projesi ve 2209-A (öğrenci araştırma projesi) bu kapsamda. Eğitilen modeli ya da görüntüleri **satılan bir ürüne koymak** bu lisanslarla olmaz. Atıflar `meta.json`'daki `kaynak` alanında ve teknik notta yazılacak.

## Sınıf eşleme

Bizim altı sınıfa eşlenmeyen her şey **atılır** (etikete yazılmaz).

| Kaynak sınıfı | → Bizim |
|---|---|
| UAVid: building | 1 bina |
| UAVid: tree | 2 agac |
| UAVid: static car, moving car | 3 arac |
| UAVid: human | 4 insan |
| UAVid: road, low vegetation, clutter | atılır |
| Semantic Drone: roof | 1 bina |
| Semantic Drone: tree | 2 agac |
| Semantic Drone: car, bicycle | 3 arac |
| Semantic Drone: person | 4 insan |
| Semantic Drone: diğer 15 sınıf (grass, wall, fence, ...) | atılır |
| VisDrone: car, van, truck, bus, motor, bicycle, tricycle, awning-tricycle | 3 arac |
| VisDrone: pedestrian, people | 4 insan |
| VisDrone: ignored regions, others | atılır |

Vinç ve kuş hiçbir açık sette yok: onlar kampüs/şantiye çekimlerinden ve Gazebo'dan gelecek.

UAVid ve Semantic Drone **segmentasyon maskesi** veriyor (her pikselin sınıfı). Bunlar maskeden çokgene çevrilip YOLO segmentasyon satırına (`sinif x1 y1 x2 y2 ...`) dönüştürülecek. Böylece elle sıfırdan etiketlemek yerine **çevir + elle kontrol et** yapılır.

## Klasör düzeni (S-5)

```
veri/
  README.md                ← depoda
  bolme.json               ← depoda
  uavid/seq1/images/000000.png    ← Drive
  uavid/seq1/labels/000000.txt    ← Drive
  uavid/seq1/meta.json            ← Drive
  semantic_drone/graz/...
  visdrone/uav0000013/...
```

`meta.json` örneği: `{"kaynak": "uavid", "irtifa_m": 50, "egim_derece": 45, "hfov_derece": null, "lisans": "CC BY-NC-SA 4.0"}`.
Bilinmeyen değer `null` yazılır, alan silinmez.

## Bölme kuralı: sahneye göre

Aynı videodan gelen kareler birbirine çok benzer. Aynı sahne hem eğitimde hem testte olursa test yalan söyler.
O yüzden bölme **sahne** (video dizisi) bazında yapılır, kare bazında değil. `bolme.json`:

```json
{"egitim": ["uavid/seq1", "uavid/seq2", "..."], "dogrulama": ["uavid/seq7"], "test": ["uavid/seq9"]}
```

Hedef oran yaklaşık %70 eğitim, %15 doğrulama, %15 test (görüntü sayısıyla). Semantic Drone tek bir alanda çekildiği için bütünüyle tek bir tarafa (eğitim) konur. `veri_kontrol.py` aynı sahnenin iki tarafta olmadığını kontrol eder.

## Pilot set: 200 görüntü

| Kaynak | Adet | Neden |
|---|---|---|
| UAVid | 100 | Eğik bina/ağaç: asıl zayıflığımız |
| Semantic Drone | 60 | Tepeden bina/ağaç |
| VisDrone | 40 | Araç/insan dengesi, kalabalık sahne |

Adımlar:
1. Lisans formlarını doldur, setleri indir (Drive'a).
2. Maskeleri bizim sınıflara çevir (dönüştürme betiği: sonraki adım).
3. CVAT ya da Label Studio'da 200 görüntüyü **elle kontrol et**: kaçan bina, birleşmiş ağaçlar, yanlış sınıf. Dışa aktarırken "YOLO" biçimi, sınıf sırası yukarıdaki tablo.
4. `python tools/veri_kontrol.py veri/` → TAMAM.

## Kontrol aracı

```bash
python tools/veri_kontrol.py --deneme   # sahte veriyle kendi kendini dene
python tools/veri_kontrol.py veri/      # gerçek klasör
```

Kontrol ettikleri: her görüntünün etiketi var mı (ve tersi), satır biçimi (kutu 5 sayı / poligon), koordinatlar 0-1, sınıf 0-5, `meta.json` alanları, `bolme.json`'da her sahne tam bir yerde mi. Sonda sınıf sayımını basar (dengesizliği görmek için).

## Karar bekleyenler

- Drive klasörü nerede, kimler erişebilir? (Lider)
- Pilot dağılımı (100/60/40) uygun mu?
- Kerem'in Gazebo etiketleri geldiğinde (Hafta 8) karışım oranı Y-4'te belirlenecek.
