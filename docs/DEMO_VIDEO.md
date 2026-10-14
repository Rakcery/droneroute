# Hocalara demo videosu (2-3 dakika)

**En kısa yol:** `docs/demo4.mp4` hazır (2 dk, 1080p). Ekran dörde bölünmüş, 4 adım aynı anda kendi kendine
uçuyor, her çeyrekte canlı test sayaçları ve Türkçe açıklama var. Yenisini üretmek için:
```
python demo_sahne.py --kaydet demo4.mp4 --sure 120
```
Bunu videonun ortasına koyup başına/sonuna aşağıdaki 1 ve 8. sahneleri kendi sesinle eklersen yeter;
gerçek fotoğraf için 5. sahneyi de ekle (o kısım gerçek YOLO modeliyle).
Aşağıdaki liste, her parçayı ayrı ayrı göstermek istersen.

Amaç: "Bu çalışıyor, ben de anlıyorum" mesajı. Her sahne tek bir fikri göstersin, sonuna kadar beklemeden kes.
Kayıt: **Win + Alt + R** (Xbox Game Bar) ya da OBS. 1080p, VS Code terminali + pencere yan yana. Sesli anlatım Türkçe, kısa.
Her komutu VS Code terminalinde `DroneRoute` klasöründeyken çalıştır.

## 1) Fikir (10 sn)
README'nin başını göster: "Kameradan engel algılayıp anlık güvenli rota hesaplayan drone. Rota algoritmaları sıfırdan yazıldı."

## 2) D* Lite: anlık yeniden planlama (25 sn)
```
python drone_dstar_lite.py
```
Fareyle rotanın önüne duvar çiz. Anlat: "Harita değişince her şeyi baştan hesaplamıyor, sadece değişen kısmı güncelliyor. Alttaki sayılar, A*'a göre kaç kat az düğüm açtığını gösteriyor."

## 3) Kamera → engel → rota, hareketli engeller (35 sn)
```
python drone_vision.py --seed 3
```
Anlat: "Kuşbakışı kamera görüntüsünden engelleri algılıyor (kırmızı), güvenlik payı bırakıyor (turuncu). Kamyonu, dönen vinç kolunu, kuşu takip edip birkaç saniye sonra nerede olacaklarını tahmin ediyor (mor)."
Birkaç kere sol tıkla yeni hedef ver. Sonra **T**'ye bas (tahmin kapanır): "Tahmin olmadan kamyon drone'un üstüne geliyor." Tekrar **T**.

## 4) İrtifa: engelin üstünden uçmak (25 sn)
```
python drone_vision.py --3d --seed 3
```
Sağ alttaki rota profilini göster: "4 irtifa katmanı var. Kamyon alçak, drone üstünden geçmek için 10 m'ye çıkıyor; vinç çok yüksek, etrafından dolaşıyor."

## 5) Gerçek drone fotoğrafında YOLO (30 sn)
```
python drone_yolo.py --resim ornekler\sokak_otobus.jpg --perspektif --irtifa 30 --egim 17 --vfov 52 --hucre-m 3 --3d --guven 0.15
```
Anlat: "Gerçek bir drone fotoğrafı. YOLO araçları, insanları ve vinci tanıyor. Kamera eğik baktığı için tespitleri geometriyle yere indirip haritaya koyuyorum, araçların boyunu da görüntüden tahmin ediyorum. İnsanların üstünden kural gereği uçulmuyor."
Hedefi **görüş alanının içinde** (siyah olmayan yerde), arabaların arkasına tıkla; drone arabaların üstünden geçmek için yükselsin.

## 6) Eğik kameralı simülasyon (20 sn)
```
python drone_vision.py --yontem yolo --model sim --perspektif --3d --seed 4
```
Anlat: "Aynı boru hattı eğik kameralı simülasyonda, hareketli engellerle birlikte. Sağ üstte kameranın gördüğü, solda haritaya indirilmiş hali."

## 7) Test edildiğini göster (15 sn, hızlandır)
```
python tests/test_yolo.py
```
Sonda `TAMAM` görünsün. Anlat: "Her adımın otomatik testi var. Binlerce karede drone hiç çarpmıyor ve D* Lite her karede A* ile aynı maliyeti buluyor."

## 8) Kapanış (15 sn)
`docs/SONRAKI_ADIMLAR.md`'yi göster: "Sırada Gazebo + PX4 ile gerçek uçuş fiziği, ardından bina/ağaç için modeli eğitmek ve LLM ile görev planlama var. ROS 2 node'ları da hazır."

## İpuçları
- Önce bir kere prova yap. Model ilk açılışta birkaç saniye yükleniyor, o kısmı kes.
- Gerçek fotoğrafta drone'u ve hedefi siyah alana koyma. Orası kameranın görmediği yer, binaları da YOLO henüz tanımıyor. Sorarlarsa dürüstçe "sıradaki iş bu" de; hocalar sınırını bilen öğrenciyi sever.
- Kendi sesinle anlat, ekrandaki her sayıyı okumaya çalışma.
