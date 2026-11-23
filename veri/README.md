# DroneRoute veri seti planı (Y-1)

YOLO'yu kendi sınıflarımızla (özellikle **bina** ve **ağaç**, yukarıdan) eğitmek için toplanacak verinin planı.
Biçim: Arayüz Sözleşmesi **S-5** (`docs/SOZLESME.md`). Kontrol: `python tools/veri_kontrol.py veri/`.

> **Görüntüler depoya girmez.** Bu klasörde depoda sadece bu README, `bolme.json` ve kontrol aracı durur.
> Görüntüler ve etiketler Drive'da: https://drive.google.com/drive/folders/1D8nIP8RB5fkKGMDiSUO8PlR-u863Q-li (ekip içi). `git status`'ta `.jpg`/`.png` görürsen ekleme.

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
| **UAVid** (uavid.nl) | 42 video dizisi (27si etiketli, dizi başına 10 kare), 4K, piksel bazında segmentasyon | Eğik, ~50 m | bina, agac, arac, insan | CC BY-NC-SA 4.0 (ticari değil, atıf, aynı lisansla paylaş) | **Pilot setin tamamı** (etiketli 270 resim indirildi) |
| **Semantic Drone Dataset** (TU Graz) | 400 eğitim görüntüsü, 20 sınıflı segmentasyon | Tepeden (nadir), 5-30 m | bina (roof), agac, arac, insan | Sadece ticari olmayan kullanım (araştırma, eğitim); atıf zorunlu; indirmek için form. **Dağıtım kısıtlı: sadece ekip içi, dışarıya paylaşılmaz** | Kullanılamıyor: dosya sunucusu ve iletişim adresi kapalı (Kasım 2026); açılırsa eklenir |
| **VisDrone2019-DET** | 8 629 görüntü, 10 sınıflı kutu | Çeşitli, tepeden/eğik | arac, insan | Resmi depoda açık lisans yok, araştırma amaçlı yayımlanmış; dağıtımlarda CC BY-NC-SA 3.0 olarak geçiyor | **Eğitime girmez:** bina/ağaç resimde var ama etiketsiz, modele "arka plan" diye öğretir. Sadece araç/insan ölçümü için (`--visdrone`) |
| **Kampüs çekimleri** | Kendi videolarımız | Eğik | hepsi | Bizim (`ic-kullanim`); yüz/plaka varsa bulanıklaştır | Y-5'te |
| **Gazebo** (Kerem, Hafta 8+) | Otomatik etiketli sim görüntüsü | Eğik kamera | hepsi | Bizim | Y-4'te |

UAVid atfı: Lyu, Vosselman, Xia, Yilmaz, Yang, *UAVid: A Semantic Segmentation Dataset for UAV Imagery*, ISPRS Journal of Photogrammetry and Remote Sensing, 2020. UAVid 42 dizi içeriyor (seq1-seq42; train, val ve etiketsiz test).

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
  _ham/uavid/uavid_v1.5_official_release_image/uavid_train/seq1/Images, Labels   ← indirildiği gibi; _ ile başlayan klasörleri veri_kontrol atlar
  _onizleme/               ← maske_cevir'in gözle kontrol resimleri
  uavid/seq1/images/000000.jpg    ← maske_cevir üretir
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

## Pilot set: UAVid, 270 görüntü

İlk plan 100 UAVid / 60 Semantic Drone / 40 VisDrone idi (PR #9). Değişti:
- **Semantic Drone** indirilemedi (sunucu ve iletişim adresi kapalı).
- **VisDrone** eğitime uygun değil: bina ve ağaçları etiketlemiyor. Etiketsiz bina/ağaç, modele "bunlar engel değil" öğretir.
- Yerine **UAVid'in etiketli bütün resimleri**: 27 dizi x 10 = 270 resim. Araç ve insan da UAVid'de bol.

| Bölme | UAVid dizileri | Resim |
|---|---|---|
| Eğitim | `uavid_train` (seq1-15, seq31-35): 20 dizi | 200 |
| Doğrulama | `uavid_val` dizileri sırayla (seq16, 18, 20, 37) | 40 |
| Test | `uavid_val` dizileri sırayla (seq17, 19, 36) | 30 |

Adımlar:
1. UAVid'i indir, `veri/_ham/uavid/` altına aç (zip'teki `uavid_v1.5_official_release_image` klasörüyle olduğu gibi). Zip Drive'da `veri/uavid/`.
2. Çevir: `python tools/maske_cevir.py` → `veri/uavid/seqN/` (S-5 düzeni, görüntüler JPG) + `veri/bolme.json` + `veri/_onizleme/` (her diziden bir resim, etiketler üstüne çizili).
3. **Gözle kontrol:** `veri/_onizleme/` altındaki 27 resme bak. Etiketler yarı saydam renkle boyalı: kırmızı bina, yeşil ağaç, pembe araç, sarı insan. **Boyanmamış yer etiketsizdir.** Çeviri doğruluğunu betik ayrıca ölçer: çokgenleri geri çizip orijinal maskeyle karşılaştırır (IoU; ilk denemede bina 0.99, ağaç 0.98, araç 0.99, insan 1.00).
4. `python tools/veri_kontrol.py veri/` → TAMAM.

Bilinen sınırlar:
- **Bizim çeviriden:** bitişik binalar tek bölge olur (maske bina bloklarını ayırmıyor); bölgenin içindeki ağaç/avlu "delik" olarak korunur (çokgene köprüyle eklenir). Sınıfa göre en küçük leke: bina/ağaç ~40x40, araç ~13x13, insan ~6x6 piksel (4K'da); altı gürültü sayılır.
- **UAVid'in kendi etiketlerinden (düzeltemeyiz):** motosiklet ve bisiklet sınıfı yok (etiketsiz kalır); kalabalıkta insanların bir kısmı etiketlenmemiş; küçük kulübe/baraka gibi yapılar çoğunlukla "clutter" (atılır). Model bu yüzden motosikleti ve kalabalığı eksik öğrenebilir; Y-3 hata analizinde ayrıca bakılacak.
- UAVid'de vinç ve kuş yok: onlar kampüs çekimleri ve Gazebo'dan (Y-4, Y-5).

## Kontrol aracı

```bash
python tools/veri_kontrol.py --deneme   # sahte veriyle kendi kendini dene
python tools/veri_kontrol.py veri/      # gerçek klasör
```

Kontrol ettikleri: her görüntünün etiketi var mı (ve tersi), satır biçimi (kutu 5 sayı / poligon), koordinatlar 0-1, sınıf 0-5, `meta.json` alanları, `bolme.json`'da her sahne tam bir yerde mi. Sonda sınıf sayımını basar (dengesizliği görmek için).

## Kararlar

- Drive klasörü açıldı, ekipte düzenleme yetkisi var.
- Pilot dağılımı 100/60/40 onaylandı (PR #9), sonra yukarıdaki sebeplerle UAVid 270'e çevrildi (bu PR).

## Açık konular

- Kerem'in Gazebo etiketleri geldiğinde (Hafta 8) karışım oranı Y-4'te belirlenecek.
- **Tepeden (nadir) görüntü yok.** UAVid'in hepsi eğik açı; tepeden bina/ağaç örneği Semantic Drone'dan gelecekti. Bu açık Gazebo görüntüleri (Y-4) ve kendi çekimlerimizle (Y-5) kapatılacak (Lider, 23 Kasım).
- **Direk / sokak lambası / tabela sınıfı yok.** Önizlemelerde fark edildi: UAVid bunları "clutter" sayıyor, bizim 6 sınıfımızda da karşılığı yok. Alçak uçuşta (5-10 m) gerçek çarpma riski. Öneri: S-5'e `6 direk` sınıfı (S-7 süreciyle, herkesin onayı). Veri UAVid'den çıkmaz; Gazebo (Y-4) ve kampüs çekimlerinden (Y-5) gelmeli.
