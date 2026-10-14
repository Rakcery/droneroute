# Sıfırdan kurulum ve test (Windows)

İki aşama var. **A** hiçbir şey kurmadan bugün yapılır, **B** Ubuntu + ROS 2 ister (ilk sefer ~30-45 dk, çoğu indirme beklemesi).

---

## A) Önce ROS'suz test (Windows, VS Code terminali)

VS Code'da DroneRoute klasörünü aç → üst menü **Terminal → New Terminal**:

```powershell
python tests/test_ros_split.py
```

Sonda `TAMAM` görmen lazım (1-2 dk). Bu, ROS'a böldüğümüz zincirin (algılama → mesaj → planlama) doğru çalıştığını ROS kurmadan kontrol ediyor. Eski demo da hâlâ çalışıyor mu bak:

```powershell
python drone_vision.py
```

---

## B) Ubuntu (WSL2) + ROS 2 Jazzy

WSL2 = Windows'un içinde gerçek bir Ubuntu. Ayrı bilgisayar ya da çift boot gerekmiyor, Ubuntu açılan bir terminal penceresi gibi çalışıyor.

### 1. Ubuntu'yu kur

1. Başlat'a **PowerShell** yaz → sağ tık → **Yönetici olarak çalıştır**.
2. Şunu yapıştır, Enter:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. Bitince **bilgisayarı yeniden başlat**.
4. Açılışta Ubuntu penceresi kendiliğinden açılır (açılmazsa Başlat → **Ubuntu 24.04**). Biraz bekler, sonra:
   - `Enter new UNIX username:` → küçük harf, boşluksuz bir ad (ör. `ryu`)
   - `New password:` → şifre. **Yazarken ekranda hiçbir şey görünmez, normal.** Bu şifreyi unutma, `sudo` her seferinde soracak.

Takılırsan:
- `Virtualization is not enabled` / `0x80370102` hatası → BIOS'ta sanallaştırma kapalı. BIOS'a gir (açılışta genelde F2 / Del), **Intel VT-x** ya da **AMD SVM** seçeneğini **Enabled** yap.
- `wsl` komutu tanınmıyor → Windows Update yap, tekrar dene.
- Kontrol: PowerShell'de `wsl -l -v` → Ubuntu-24.04 yanında **VERSION 2** yazmalı.

### 2. ROS 2 Jazzy'yi kur

Bundan sonraki her şey **Ubuntu penceresinde**. Blokları tek tek kopyala, Ubuntu penceresine **sağ tıkla** (yapıştırır), Enter. Şifre sorarsa Ubuntu şifren.

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
```

```bash
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-jazzy-ros-base ros-jazzy-rqt-image-view ros-jazzy-rviz2 python3-colcon-common-extensions python3-opencv python3-numpy
```

(Bu sonuncusu uzun sürer, birkaç yüz MB iniyor.)

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
ros2 --help
```

Son komut bir yardım listesi basıyorsa ROS 2 kuruldu. ✅

### 3. Projeyi ROS'a tanıt (bir kere)

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ln -s /mnt/c/Users/steam/Desktop/Project/DroneRoute DroneRoute
ls DroneRoute
```

`ls` çıktısında `drone_vision.py`, `package.xml` vs. görmen lazım (Windows'taki klasörün ta kendisi; Ubuntu C: diskini `/mnt/c` olarak görüyor).

```bash
cd ~/ros2_ws && colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

`Summary: 1 package finished` görmelisin.

### 4. Çalıştır ve test et

```bash
ros2 launch droneroute droneroute.launch.py
```

Windows'ta bir pencere açılır (ilk sefer birkaç saniye sürebilir). Neye bakacaksın:

- [ ] Şantiye sahnesi, sarı drone, yeşil hedef, mavi rota görünüyor
- [ ] Drone hedefe gidiyor, varınca yeni hedef seçiyor
- [ ] Sol tık → hedef oraya taşınıyor, sağ tık → drone oraya ışınlanıyor
- [ ] Kamyon yoluna hedef koy: kamyon gelince drone kenara çekilip sonra geri dönüyor
- [ ] Mor hücreler (hareketli engelin geleceği yer) kamyonun ve vinç kolunun önünde

Kapatmak için Ubuntu terminalinde **Ctrl+C**.

Node'ların gerçekten ayrı çalıştığını görmek için: launch açıkken **ikinci bir Ubuntu penceresi** aç (Başlat → Ubuntu 24.04):

```bash
ros2 node list                  # /camera /detection /planner /viz
ros2 topic hz /obstacles/grid   # saniyede ~25-30 mesaj (Ctrl+C ile dur)
ros2 topic echo /plan/status    # drone'un o anki durumu
```

Karşılaştırma (kamyonun drone'u ezdiğini görürsün):

```bash
ros2 launch droneroute droneroute.launch.py tahmin:=false
```

---

## Sonra ne değişirse

- `.py` dosyasını VS Code'da değiştirdin → sadece launch'ı kapatıp tekrar aç.
- Yeni dosya eklendi, `setup.py` ya da launch dosyası değişti → önce `cd ~/ros2_ws && colcon build --symlink-install`.

## Sık sorunlar

| Belirti | Çözüm |
|---|---|
| `Package 'droneroute' not found` | `source ~/ros2_ws/install/setup.bash`, olmadıysa adım 3'teki build'i tekrar et |
| `option --editable not recognized` | `pip install --upgrade setuptools` yapılmış. `pip uninstall setuptools` ile geri al |
| Pencere açılmıyor, `ekran yok` uyarısı | PowerShell'de `wsl --update`, sonra `wsl --shutdown`, Ubuntu'yu tekrar aç |
| `ls DroneRoute` boş / bulunamadı | Klasör yolu farklı. Windows'ta klasöre sağ tık → Yolu kopyala; `C:\a\b` → `/mnt/c/a/b` olarak `ln -sfn` ile tekrar bağla |
| Webcam açılmıyor | WSL2 USB kamerayı görmüyor; ROS'ta sim ya da `kaynak:=video` kullan |
