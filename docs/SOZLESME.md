# Arayüz Sözleşmesi v1.0 (14 Ekim 2026)

Bu blok dört kişisel notta da (Lider, Enes, Yağmur, Kerem) **kelimesi kelimesine aynıdır**. "Sözleşme", iki parçanın birbirine hangi veriyi hangi biçimde vereceğinin yazılı hâlidir. Biçim sabit kaldıkça herkes kendi parçasını öbürünü beklemeden geliştirir; karşı taraf henüz hazır değilse aynı biçimde **sahte veri** (mock: gerçeğinin yerine geçici olarak kullanılan, biçimi aynı uydurma veri) kullanılır. Sahte verinin kodu `sahte_veri.py` dosyasındadır (aşağıda tam metni var).

## S-0 · Ortak dil

| Kavram | Değer ve anlamı |
|---|---|
| Izgara | 40 sütun (`COLS`) × 28 satır (`ROWS`), `drone_dstar_lite.py` başında |
| Hücre | `(x, y)` demeti. x = sütun (0-39, soldan sağa), y = satır (0-27, **yukarıdan aşağı**, görüntüdeki gibi) |
| Tablo | 28 × 40 NumPy dizisi. Okurken `tablo[y, x]` (önce satır, sonra sütun) |
| Kare | Kameradan gelen bir görüntü. Zaman birimi "kare"dir (Python simülasyonunda saniyede ~30) |
| Hücre boyu `res` | Python simülasyonu 0,5 m (`codec.RESOLUTION`). Gazebo'da 2,0 m (`hucre_m`). Dönüşüm fonksiyonları `res`'i parametre alır |
| Katman (3D) | `ALTS = (5, 10, 15, 20)` metre; katman numarası `z = 0..3` |
| `inf` (sonsuz) | "Bilinmiyor" ya da "asla". Boy bilinmiyorsa `inf` = üstünden uçulmaz |
| Güvenli taraf | Kodlamada yuvarlama hep çarpışmayı önleyen yöne: boy **yukarı**, tti **aşağı** (erken) yuvarlanır |

## S-1 · Algı → Planlama (Yağmur → Enes) · mevcut, değişmiyor

```python Python arayüzü (drone_vision.Perception)
raw, moving_tti = perc.process(frame, height=None)   # bir kare işle
perc.height          # 28x40 float: raw hücrelerdeki engelin boyu (m); boş = 0, bilinmiyor = inf
perc.moving_height   # 28x40 float: moving_tti'deki hücreye gelecek engelin boyu (m)
```

| Ad | Tip | Anlamı |
|---|---|---|
| `raw` | 28×40 bool | Zaman filtresinden geçmiş, **şu an** dolu hücreler |
| `moving_tti` | 28×40 float | Hareketli engel bu hücreye **kaç kare sonra** giriyor; gelmiyorsa `inf` |
| `height` | 28×40 float | Engel boyu (m); boş 0; bilinmiyor `inf` |
| `moving_height` | 28×40 float | Gelecek hareketli engelin boyu (m) |

ROS 2 topic'leri (hepsi `nav_msgs/OccupancyGrid`, aynı kamera zaman damgası, kodlama `droneroute_ros/codec.py`): `/obstacles/grid` (100 dolu, 0 boş) · `/obstacles/tti` (-1 gelmiyor, 0..100 = 5 × kare) · `/obstacles/height` ve `/obstacles/tti_height` (metre, yukarı yuvarlanır, -1 bilinmiyor). Planlama, güvenlik payı ve duvar/ceza kararı **algıda değil planlamada** verilir.
Sahte veri: `sahte_veri.sahte_algi(kare)` → `(raw, height, moving_tti, moving_height)`.

## S-2 · Planlayıcı arayüzü (Enes → Kerem, Lider) · mevcut + E-2 eklemesi

```python Python arayüzü (drone_vision.GridPlanner ve drone_plan3d.GridPlanner3D)
gp.update(raw, moving_tti)                          # 2D: bir karelik algıyı uygula
gp.update(raw, height, moving_tti, moving_height)   # 3D: boylarla birlikte
gp.set_goal((x, y)); gp.set_drone((x, y))           # hedefi / drone'u taşı
gp.step_drone()            # rotada bir düğüm ilerle; hedefe vardıysa True
gp.set_nofly([((x, y), r), ...])   # YENİ (E-2, Hafta 5): yasak daireler, merkez hücre + yarıçap (hücre)
gp.path    # 2D: [(x, y), ...]   3D: [(x, y, z), ...]   rota yoksa None
gp.drone, gp.z, gp.goal, gp.status   # konum (hücre), katman, hedef, durum yazısı (ASCII)
```

ROS 2: girdi `/obstacles/*` (S-1), `/goal_pose`, `/initialpose`. Çıktı `/plan/path` (`nav_msgs/Path`; her poz hücre merkezinin map koordinatı (metre), 3D'de `z = ALTS[z]` metre), `/plan/costmap`, `/drone/pose`, `/plan/goal`, `/plan/status`.
Sahte veri: `sahte_veri.sahte_rota_hucre()` → `[(x, y, z), ...]`.

## S-3 · Uçuş rotası ve takip (Enes → Kerem) · YENİ

Uçuş rotası = **nokta listesi** `[(x, y, z, v), ...]`: x, y, z metre (map çerçevesi, S-4), v hız (m/s). İlk ve son noktada `v = 0`.

```python drone_smooth.py ve drone_follow.py (Enes yazar, Kerem px4_bridge içinden çağırır)
shortcut(path, grid) -> list                  # E-1: hücre rotası -> kısaltılmış hücre rotası (alt küme)
smooth_path(cells, grid, res) -> [(x, y, z)]  # E-3: yumuşak eğri, metre; güvenlik payına taşmaz
speed_profile(points) -> [(x, y, z, v)]       # E-3: hız planı, uçuş rotası
pp = PurePursuit(lookahead=2.0)               # E-4: takip denetleyicisi (ileri bakış, metre)
pp.set_path(waypoints)                        # uçuş rotasını ver
pp.target((x, y, z)) -> (x, y, z, v) | None   # drone konumu -> gidilecek nokta; None = rota yok
pp.done                                       # hedefe vardıysa True
```

`px4_bridge` bu fonksiyonları **saniyede 20 kez** (her 0,05 sn) çağırır. Hepsi ROS'suz, saf Python.
Sahte veri: `sahte_veri.SahteTakip` (aynı üç metot), `sahte_veri.sahte_ucus_rotasi()`.

## S-4 · Konum, koordinat çerçeveleri, güvenlik (Kerem → herkes) · YENİ

| Konu | Karar |
|---|---|
| map çerçevesi (ENU) | Orijin ızgaranın sol alt köşesi. x doğu (sağa), y kuzey (yukarı), z yukarı. Metre = hücre × `res`. `codec.cell_to_xy` / `codec.xy_to_cell` |
| PX4 çerçevesi (NED) | Orijin **kalkış noktası**. x kuzey, y doğu, z **aşağı** |
| Kalkış noktası | Hücre `KALKIS_HUCRE = (1, 14)`; map'te `KALKIS_XY = cell_to_xy(KALKIS_HUCRE, res)`. Gazebo dünyasının orijini = kalkış noktası |
| Dönüşüm | `enu_to_ned(x, y, z) = (y - KALKIS_Y, x - KALKIS_X, -z)` · `ned_to_enu(n, e, d) = (e + KALKIS_X, n + KALKIS_Y, -d)` · yön: `yaw_ned = π/2 - yaw_enu` |
| `/px4/pose` | `geometry_msgs/PoseStamped`, map çerçevesi, ENU, metre, saniyede ≥ 20. Yayınlayan: `px4_bridge` |
| `/safety/state` | `std_msgs/String`: `NORMAL` · `BEKLE` · `EVE_DON` |
| Güvenlik kuralı | Kamera 1 sn gelmezse → `BEKLE` (havada dur). Rota 1 sn gelmezse ya da boşsa → `BEKLE`. 10 sn `BEKLE` → `EVE_DON` (PX4 eve dönüş) |
| `planner_node` | Yeni parametre `konum:=sim` (bugünkü taklit drone) ya da `konum:=px4` (`/px4/pose`'u kullanır) |
| Kamera, v1 | Gazebo'da **sabit gözlem kamerası**: kalkış noktasına göre (37, 1, 46) m'de, tam aşağı bakar, 880×616 piksel, yatay görüş açısı 82°. Görüntüsü 80 m × 56 m = 40 × 28 hücre × 2 m. Izgara dünyaya sabit, bugünkü kodla aynı varsayım |
| Kamera, v2 | Hafta 9-10 (E-5 ile): drone'un altındaki eğik kamera + hafızalı dünya haritası. Ayrıntısı Sözleşme v2'de yazılacak |

Sahte veri: `sahte_veri.sahte_konum(t)` (ENU), `sahte_veri.enu_to_ned / ned_to_enu` (referans formül; gerçeği `codec.py`'de, K-4).

## S-5 · Etiketli görüntü verisi (Kerem ↔ Yağmur) · YENİ

| Konu | Karar |
|---|---|
| Sınıflar (sıra = numara) | `0 vinc · 1 bina · 2 agac · 3 arac · 4 insan · 5 kus` (`drone_yolo.CLASSES` ile aynı adlar) |
| Klasör | `veri/<kaynak>/<sahne>/images/<ad>.jpg` (ya da `.png`) + `veri/<kaynak>/<sahne>/labels/<ad>.txt` + `meta.json` |
| Etiket satırı | Kutu: `sinif xc yc w h` · Segmentasyon: `sinif x1 y1 x2 y2 ...`. Hepsi 0-1 arasına ölçeklenmiş (YOLO biçimi) |
| `meta.json` | `{"kaynak", "irtifa_m", "egim_derece", "hfov_derece", "lisans"}` |
| Gazebo etiketi | Gazebo `Label` eklentisinde etiket numarası = sınıf + 1 (0 = arka plan) |
| Nerede durur | Görüntüler **depoya girmez** (Drive ya da GitHub Release). Depoda sadece `veri/README.md` (kaynak + lisans listesi) |

Sahte veri: `sahte_veri.sahte_etiket_klasoru(yol)`.

## S-6 · Ölçüm sonuçları (herkes → Lider) · YENİ

Her ölçüm bir CSV satırıdır: `sonuclar/<kart>_<kisa-ad>.csv` (ör. `sonuclar/e-1_kisaltma.csv`). Başlık satırı tam olarak: `kart,deney,yontem,sahne,metrik,deger,birim`. UTF-8, ondalık ayırıcı nokta. Yazmak için `sahte_veri.sonuc_yaz(yol, satirlar)`. Lider `tools/rapor_tablosu.py` ile hepsini tek tabloda birleştirir (rapor ve 2209-A için).
Y-3'ün Enes'e özeti: `veri/algi_ozet.json` = `{"model", "en_kucuk_nesne_px", "en_kucuk_nesne_m", "onerilen_pay_hucre", "not_"}`.

## S-7 · Sözleşme nasıl değişir

1. Tek kaynak: depoda `docs/SOZLESME.md` + `sahte_veri.py` (Lider, L-5). Bu not ile depo çelişirse **depo geçerlidir**.
2. Değişiklik = birleştirme isteği (PR), etiketi `sozlesme`, başlığı "Sözleşme v1.1: ...". Lider yazar (S), diğer üçü onaylar (O). Genel rehber §6.1'deki RACI tablosu.
3. Ekleme (yeni alan, yeni topic) geriye uyumludur → v1.x. Var olan bir şeyin anlamını değiştirmek → v2, herkesin onayı ve `sahte_veri.py` güncellemesi **aynı PR'da**.
4. Yeni bir `drone_*.py` modülü eklenince `setup.py` içindeki `py_modules` listesine de eklenmeli (Kerem'e haber ver), yoksa ROS 2 node'ları onu bulamaz.

Sahte verinin tam kodu: depo kökündeki `sahte_veri.py`.
