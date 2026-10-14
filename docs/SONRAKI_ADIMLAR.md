# DroneRoute: Sonraki adımlar

Şu an elimizde: kameradan engel algılama (renk/kenar/YOLO), hareketli engel takibi ve tahmini, 2D ve 4 katmanlı 3D D* Lite, eğik kameradan zemine indirme, ROS 2 node'ları. Hepsi **simülasyonda ve tek kare üzerinde** doğru çalışıyor. Bundan sonrası, bu beyni gerçek bir uçuş sistemine bağlamak ve gerçek dünyanın getirdiği belirsizliği yönetmek.

Sıralama önerim: **A → B → C**, D ve E ise arada istediğin zaman yapılabilir.

---

## A) Gazebo + PX4 SITL: gerçek uçuş dinamiği (yol haritasındaki 4. adım)

**Neden:** Şu an drone "her 4 karede bir hücre zıplıyor". Gerçek bir drone ivmeleniyor, dönerken savruluyor, rüzgârda kayıyor, irtifa değiştirmesi zaman alıyor. Kamera da onunla birlikte hareket ediyor. Bunların hiçbiri şu an yok.

**Ne kurulacak:**
- **Ubuntu 24.04 (WSL2) + ROS 2 Jazzy** (KURULUM_WSL.md'de hazır).
- **Gazebo Harmonic**: fizik simülatörü. İçine bir şantiye dünyası (`.sdf`): binalar, ağaçlar, hareket eden bir kamyon, dönen bir vinç.
- **PX4 SITL**: gerçek drone'larda çalışan uçuş yazılımının bilgisayarda çalışan hali. Motorları, dengeyi, konum kontrolünü o yapıyor; biz ona sadece "şu noktaya git" diyeceğiz.
- **Micro XRCE-DDS köprüsü**: PX4 ile ROS 2'nin konuşması için (`/fmu/in/trajectory_setpoint`, `/fmu/out/vehicle_odometry` gibi topic'ler).

**Bizim yazacağımız kısım:**
1. **Kamera node'u yerine Gazebo kamerası.** Drone'un altına eğik bakan bir RGB kamera, istersek derinlik kamerası. `detection` node'u değişmeden `/camera/image_raw` dinliyor, `--perspektif` modu tam bu iş için yapıldı.
2. **Hareketli kamera (ego-motion).** Şu an kamera sabit varsayılıyor. Drone hareket edince her kare farklı yerden çekiliyor. PX4'ün verdiği konum ve yön (odometri) ile her karenin tespitleri **dünya koordinatına** çevrilecek. Grid artık "kameranın önü" değil, sabit bir **dünya haritası** olacak (ör. 200 m x 200 m, drone içinde geziyor). Homografi her karede kameranın o anki irtifa/eğimiyle yeniden hesaplanacak (kodda `Camera` ve `GroundProjector` bunun için ayrı sınıflar).
3. **Hafızalı harita.** Kameranın şu an görmediği yerdeki engeller unutulmamalı. Hücre başına "son ne zaman, ne kadar emin gördük" tutulacak: görülen hücre güncellenir, görülmeyen hücre yavaş yavaş "bilinmiyor"a döner (sabit bina kalır, kamyon zamanla silinir).
4. **Planlayıcı → uçuş kontrolcüsü.** D* Lite'ın hücre rotası doğrudan uçulamaz (köşeli). Arada:
   - **Rota yumuşatma:** gereksiz ara noktaları atan "görüş hattı" kısaltması (iki nokta arası engelsizse aradakileri sil), sonra B-spline ile yumuşatma.
   - **Hız profili:** köşede yavaşla, düzde hızlan, irtifa değişimini yatay hareketle birleştir.
   - **Takip:** her 50 ms'de PX4'e bir sonraki ara nokta (`TrajectorySetpoint`). Bunu basit bir "havuç takibi" (pure pursuit) ile kendimiz yazarız, kurallara uygun.
5. **Güvenlik katmanı.** Rota yoksa veya algılama durduysa (kamera donması) drone yerinde asılı kalsın; uzun sürerse eve dönsün ya da insin (PX4 failsafe modları ile).

**Başarı ölçütü:** Gazebo'daki şantiyede drone kalkıyor, kamyonun önünden kaçıyor, alçak engelin üstünden geçiyor, vincin etrafından dolaşıp hedefe iniyor. Aynı testleri (çarpma = 0) Gazebo'da tekrar koşuyoruz.

---

## B) Algılamayı gerçek dünyaya hazırlamak

**1. Bina/ağaç için ince ayar (fine-tune).** YOLOE araç/insan/vinçte iyi ama binayı yukarıdan neredeyse hiç bulmuyor. Çözüm: küçük bir eğitim.
- Veri: Gazebo'dan otomatik etiketli görüntü (simülatör her pikselin ne olduğunu biliyor, etiket bedava) + gerçek drone görüntüsü içeren açık veri setleri (bina/ağaç segmentasyonu olan hava görüntüleri).
- Model: YOLO11n-seg ya da YOLOE'nin kendisi, birkaç bin görüntüyle. Bir GPU'da (Colab'ın ücretsiz GPU'su yeter) birkaç saat.
- Bizim koddaki sözlük (`vinc/bina/agac/...`) aynen kalıyor; `--model yeni.pt` ile takılıyor.

**2. Segmentasyonla daha doğru taban.** Şu an eğik kamerada engelin tabanını kutunun alt kenarından buluyoruz, arkaya doğru uzunluğu tahmin. Segmentasyon maskesinin her sütundaki en alt pikseli zemine değen gerçek kontur. Taban poligonu çok daha doğru çıkar.

**3. Gerçek derinlik.** Üç seçenek, ucuzdan pahalıya:
- **Monoküler derinlik ağı** (ör. Depth Anything): tek kameradan göreli derinlik. Bizim geometrik tahminimizle ölçeklenip birleştirilir.
- **Stereo kamera** (OAK-D, ZED): gerçek metre cinsinden derinlik. `/camera/height` topic'i zaten var, oraya bağlanır.
- **Hareketten derinlik:** drone hareket ettikçe aynı noktayı iki farklı yerden görüyoruz, üçgenleme ile mesafe. Klasik algoritma, kendimiz yazabiliriz, sunumda da güzel durur.

**4. Belirsizlik.** Her tespitin bir güveni var (0.25 … 0.9). Şu an eşik üstü = kesin engel. Daha iyisi: hücre başına olasılık (log-odds occupancy grid): her karede tespit gelince artır, gelmeyince azalt. Tek karelik yanlış tespit rotayı bozmaz, gerçek engel birkaç karede kesinleşir. Zaman filtremizin (K_ON/K_OFF) olasılıklı hali.

---

## C) Planlamayı güçlendirmek

**1. Vinç kolunun altından geçmek.** Şu an YOLO'da vinç "sonsuz boy". Gerçekte 20 m'de uçan drone 35 m'deki kolun altından geçebilir, sadece kule ve kanca halatı tehlikeli. Derinlik ya da segmentasyonla kule ile kol ayrılırsa, kol için "alt sınır–üst sınır" (z0–z1) aralığı tutulur. Planlayıcıda "engel boyu" tek sayı yerine aralık olur: bir katman, ancak engelin aralığıyla kesişiyorsa kapalı sayılır. Simülasyonda bu bilgi zaten var (`SimScene.objects()` her parçanın z0/z1'ini veriyor).

**2. Hedef kapalıyken akıllı bekleme.** Ekran görüntüsündeki durum: hedef hücresinde vinç ve kuş var, ikisi de "üstünden uçulmaz" sınıfı, yani hiçbir irtifada hedefe girilemiyor. Drone şu an olduğu yerde bekliyor. Daha iyisi: hedefe en yakın ulaşılabilir hücreye gidip orada beklemek; vinç kolu döndüğü için tahmin "hedef X kare sonra açılacak" diyorsa o zamana göre varış planlamak.

**3. Zaman boyutlu planlama.** Şu an hareketli engeller "yakında gelecek hücre = duvar/pahalı" olarak giriyor. Bir üst seviye: düğüm = (x, y, z, **t**), yani "şu hücrede şu anda olmak". Kamyonun arkasından geçmeyi ya da vinç kolunun geçmesini bekleyip sonra geçmeyi optimal olarak bulur. Klasik karşılığı: Safe Interval Path Planning (SIPP), sıfırdan yazılabilir ve D* Lite'la karşılaştırması iyi bir sunum konusu.

**4. Enerji ve rüzgâr.** Kenar maliyetine pil tüketimi (tırmanma pahalı, zaten var) ve rüzgâr yönü eklenebilir; rüzgâra karşı uçmak pahalı.

**5. Daha büyük harita.** 40x28 grid küçük. Gerçek bir şantiye için ya grid büyütülür (D* Lite artımlı olduğu için iyi ölçekleniyor), ya da iki seviyeli plan: kaba grid'de uzun rota, ince grid'de yakın çevre.

---

## D) Üst seviye görev planlama (LLM, isteğe bağlı)

Kuralımıza uygun: LLM **rota hesaplamıyor**, sadece görevi parçalıyor.

- Kullanıcı: "Kuzeydeki üç vinci denetle, sonra B blokun çatısını çek, pil %30'un altına düşerse dön."
- LLM bunu yapılandırılmış ara hedeflere çevirir: `[{tip: "denetle", hedef: "vinc_1", irtifa: 20}, ...]`. Hangi vinçlerin nerede olduğunu bizim algılama haritamızdan okur.
- Her ara hedefe rotayı D* Lite hesaplar, LLM'in çıktısı doğrulanır (hedef haritada var mı, irtifa sınırda mı). Geçersizse reddedilir.
- Sunumda güzel bir demo: konuşma kutusuna görev yaz, drone Gazebo'da uygulasın.

---

## E) Sunum ve ölçüm

- **Karşılaştırma tablosu:** renk vs. kenar vs. YOLO vs. hibrit (kesinlik/duyarlılık), 2D vs. 3D (rota uzunluğu, tırmanma sayısı), D* Lite vs. her karede A* (açılan düğüm, ms). Testler bu sayıların çoğunu zaten basıyor.
- **Video:** simülasyon ekran kaydı, sonra Gazebo kaydı, en son (varsa) gerçek drone görüntüsünde YOLO + kuşbakışı grid.
- **Gerçek uçuş (en son):** küçük bir drone (ör. PX4 destekli bir geliştirme kiti) ile sadece kamera kaydı alıp algılamayı yerde test etmek, sonra açık alanda, kurallara uygun, düşük irtifada.

---

## Kısa özet: bir sonraki somut iş

1. Ubuntu + ROS 2 + Gazebo + PX4 SITL kurulumu (WSL2).
2. Gazebo'da şantiye dünyası + eğik kameralı drone.
3. Kamera node'u yerine Gazebo kamerası, planlayıcı çıktısını PX4'e ara nokta olarak gönderen yeni bir `px4_bridge` node'u.
4. Kamera hareket edince dünya koordinatlı, hafızalı harita.

Bunlar bitince drone ilk kez "gerçek" fizikle uçmuş olacak; ardından B ve C'deki iyileştirmeler.
