# Y-3 · Algılama hata analizi

Model: `kendi_y2.pt` (Y-2 ince ayar, yolo11n-seg, 1280 px). Test seti: UAVid seq17, seq19, seq36 (30 resim, eğitimde hiç görülmedi), eğik kamera, ~50 m.
Araç: `python tools/hata_analizi.py` → `sonuclar/y-3_hata.csv`, `veri/algi_ozet.json`, örnekler `veri/_hata_ornekleri/` (depoya girmez).

Nesne nesne analiz (IoU ≥ 0.5) **araç ve insan** için. Bina ve ağaç UAVid'de bitişik bölgeler hâlinde etiketli, model ise tek tek buluyor; kutu eşleme orada yanıltır. Onların alan bazlı sonucu Y-2'de (bina duyarlılık 0.91, ağaç 0.74).

## 1. Sınıf başına

| Sınıf | Gerçek nesne | Bulundu | Duyarlılık | Yanlış alarm | Kesinlik |
|---|---|---|---|---|---|
| araç | 247 | 135 | 0.55 | 61 | 0.69 |
| insan | 57 | 7 | 0.12 | 7 | 0.50 |

Araç duyarlılığı nesne bazında 0.55, alan bazında 0.73 (Y-2): büyük araçlar bulunuyor, küçükler kaçıyor. Kesinlik gerçekte bundan yüksek (bkz. §4: "yanlış alarm"ların bir kısmı etiketlenmemiş gerçek araç).

## 2. Asıl etken: boy

| Boy (4K piksel, √alan) | Modelin gözünde (1280'e küçültülmüş) | araç | insan |
|---|---|---|---|
| 0-16 | 0-5 px | 0.00 (5) | 0.00 (18) |
| 16-24 | 5-8 px | 0.00 (32) | 0.10 (10) |
| 24-32 | 8-11 px | 0.12 (17) | 0.25 (12) |
| 32-48 | 11-16 px | 0.36 (33) | 0.30 (10) |
| 48-64 | 16-21 px | 0.59 (34) | 0.00 (4) |
| 64-96 | 21-32 px | 0.73 (60) | 0.00 (1) |
| 96+ | 32+ px | 0.86 (66) | 0.00 (2) |

(parantez içi: o aralıktaki nesne sayısı)

**Sonuç:** 4K resimde ~48 pikselden (modelin gözünde ~16 piksel) küçük nesnelerin yarısından azı bulunuyor. Test setindeki insanların %70'i 32 pikselin altında; insan duyarlılığının 0.12'de kalmasının ana sebebi bu. 48 px üstündeki insan sayısı (7) yorum yapmaya yetmiyor.

## 3. Diğer etkenler

| Etken | Grup | Duyarlılık (araç + insan) |
|---|---|---|
| uzaklık | yakın (resmin alt üçte biri) | 0.71 (51) |
| uzaklık | orta | 0.48 (124) |
| uzaklık | uzak (üst üçte bir) | 0.36 (129) |
| keskinlik | bulanık | 0.48 (100) |
| keskinlik | orta | 0.48 (104) |
| keskinlik | keskin | 0.44 (100) |
| kadraj | kesik (kenara değiyor) | 0.45 (11) |
| kadraj | tam | 0.47 (293) |

- **Uzaklık** etkili, ama büyük ölçüde boyun başka bir yüzü: eğik kamerada uzaktaki nesne küçük görünüyor.
- **Bulanıklık** belirgin bir etken değil (0.44-0.48). Not: keskinlik ilk denemede nesnenin kendi kutusunda ölçülmüştü ve minik kutular yapay olarak "keskin" çıkıyordu (keskin 0.24 gibi yanıltıcı bir sonuç). Şimdi her nesne için aynı boy (96x96) pencerede ölçülüyor.
- **Kadrajdan kesik** olmak fark yaratmıyor (az örnek: 11).

## 4. Hata örnekleri (28 örnek, `veri/_hata_ornekleri/`)

| # | Tür | Sınıf | Gözlem | Sebep grubu |
|---|---|---|---|---|
| 00-04 | kaçırıldı | araç | 16-28 px, uzakta park etmiş araçlar | çok küçük / uzak |
| 05 | kaçırıldı | araç | 58 px, gölgede koyu araç | gölge |
| 06 | kaçırıldı | araç | 90 px, çatının yanında, kısmen örtülü | kısmen görünür |
| 07, 08, 10 | kaçırıldı | insan | 20-29 px, açık zeminde net görünen yürüyen insan | çok küçük |
| 09, 12 | kaçırıldı | insan | şezlongda uzanmış insanlar | alışılmadık duruş |
| 11 | kaçırıldı | insan | 17 px, motosiklet üstünde | çok küçük + başka nesneyle iç içe |
| 13 | kaçırıldı | insan | 167 px kutu, içinde insan görünmüyor | etiket hatası (UAVid) |
| 14, 16, 17, 18 | yanlış alarm | araç | minibüs, otomobil, arazi aracı: **gerçek araçlar**, UAVid etiketlememiş | etiketsiz gerçek nesne |
| 15 | yanlış alarm | araç | çatı penceresi | benzer görünüm |
| 19 | yanlış alarm | araç | bahçe şemsiyesi / tente | benzer görünüm |
| 20 | yanlış alarm | araç | motosiklet / üç teker kümesi | UAVid'de sınıfı yok (clutter) |
| 21, 22, 24 | yanlış alarm | insan | yerde siyah-beyaz dikdörtgen nesne (bariyer / tabela tabanı) | benzer görünüm |
| 23 | yanlış alarm | insan | çimde çim biçme makinesi / sandalye | benzer görünüm |
| 25 | yanlış alarm | insan | **gerçek bir insan**, etiketsiz | etiketsiz gerçek nesne |
| 26, 27 | yanlış alarm | insan | çöp kutuları (yukarıdan insan gibi) | benzer görünüm |

Gruplanmış: **kaçırılanlar** (14): çok küçük/uzak 9, alışılmadık duruş 2, gölge 1, kısmen görünür 1, etiket hatası 1. **Yanlış alarmlar** (14): benzer görünüm 8, etiketsiz gerçek nesne 5, sınıfı olmayan nesne (motosiklet) 1.

Yani yanlış alarmların bir kısmı aslında doğru; ölçtüğümüz kesinlik gerçeğinden düşük. Planlama açısından "benzer görünüm" yanlış alarmları zararsız (drone gereksiz dolaşır), ama altın kuralımıza göre asıl sorun küçük nesneleri kaçırmak.

## 5. Enes'e özet (`veri/algi_ozet.json`)

- **En küçük güvenilir nesne:** 4K resimde ~48 px (√alan), modelin gözünde ~16 px. UAVid'in ~50 m eğik çekiminde yaklaşık **3.2 m** (aynı resimdeki araçlar 4.5 m sayılarak; yaklaşık bir değer, kesin ölçek için kalibrasyon lazım, Y-5).
- **Konum hatası:** bulunan nesnelerde kutu kenarı medyan 0.33 m, %95'te 1.32 m kayıyor. Gazebo hücresi 2 m olduğuna göre **1 hücre güvenlik payı yeterli** (bugünkü değer).
- **Ama:** pay sadece *bulunan* nesneyi korur. 3 m'den küçük olup hiç bulunmayan nesneye (insan ~0.5 m, direk, tabela) pay işe yaramaz. Alçak uçuşta (5-10 m) bunlar için ayrı bir kural gerekir: ör. insan olabilecek bölgelerde (kaldırım, meydan) en alçak katmanı kapatmak ya da bu nesneleri boy aralığına göre "bilinmiyor = sonsuz" saymak. Enes ve Lider'le konuşulacak.

## 6. Ne yapılabilir? (öneriler)

1. **Parçalı tahmin (tiling):** 4K resmi 1280'lik 6 parçaya bölüp her birini ayrı tahmin etmek. İnsan modelin gözünde 8 px yerine ~25 px olur; tablodaki boy eşiğine göre insan duyarlılığı belirgin artmalı. Kod değişikliği küçük (sadece `drone_yolo.py`'nin tahmin kısmı), CPU süresi ~6 kat. En ucuz ve en etkili adım.
2. **Daha çok insan örneği:** pilot sette 5495 insan var ama çoğu çok küçük. Daha alçaktan çekilmiş veri (kampüs çekimleri, Y-5; Gazebo, Y-4).
3. **Etiket temizliği:** test setindeki etiketsiz gerçek araçlar/insanlar ölçümü bozuyor; Y-4'te Gazebo'nun otomatik etiketleri bu sorunu taşımaz.
4. **Direk / motosiklet sınıfı** (açık konu, `veri/README.md`).
