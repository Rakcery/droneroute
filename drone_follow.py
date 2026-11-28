"""drone_follow.py: uçuş rotasını takip (E-4, S-3). ROS'suz, saf Python."""
import math                                   # uzaklık, karekök

from drone_smooth import V_MAX                # hız planının en yüksek hızı (m/s): havuç bu hızda tam lookahead uzakta

LOOKAHEAD = 2.0                               # havuç drone'un kaç metre önünde (m)
VARIS = 0.3                                   # hedefe bu kadar yaklaşınca "vardı" say (m)
ARAMA = 2.0                                   # en yakın noktayı ararken lookahead'in kaç katı ileriye bakılır (kat)
L_MIN = 0.5                                   # yavaşken havucun en yakın mesafesi (m)
V_MIN = 0.3                                   # en düşük takip hızı: kalkışta ve hedefe yaklaşırken (m/s)

def izdusum(p, a, b):
    """p noktasının a-b parçası üzerindeki en yakın yeri: t (0 = a'da, 1 = b'de)."""
    ab = [b[k] - a[k] for k in range(3)]                     # parçanın yönü (x, y, z)
    boy2 = sum(c * c for c in ab)                            # parça boyunun karesi
    if boy2 == 0:                                            # a ile b aynı nokta
        return 0.0                                           # a'nın kendisi
    t = sum((p[k] - a[k]) * ab[k] for k in range(3)) / boy2  # gölgenin parça üzerindeki oranı
    return min(1.0, max(0.0, t))                             # parçanın dışına taşmasın

def ara_nokta(a, b, t):
    """a ile b arasında t oranındaki nokta (x, y, z, v): hız da aradan alınır."""
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(4))   # dört değeri de doğrusal karıştır

class PurePursuit:
    """S-3: set_path(waypoints), target(pos) -> (x, y, z, v) | None, done."""

    def __init__(self, lookahead=LOOKAHEAD):
        self.lookahead = lookahead            # ileri bakış mesafesi (m), ayarlanabilir
        self.wps = []                         # rota noktaları [(x, y, z, v), ...]
        self.i = 0                            # drone'un şu an bulunduğu parça: wps[i] -> wps[i+1]
        self.done = False                     # hedefe vardı mı

    def set_path(self, waypoints):
        """Yeni rota ver: baştan başla."""
        self.wps = list(waypoints)            # rotayı kopyala (dışarıda değişirse bizi etkilemesin)
        self.i = 0                            # ilk parçadan başla
        self.done = len(self.wps) == 0        # boş rota = yapacak iş yok
        self.golge = self.wps[0] if self.wps else None   # drone'un rotadaki gölgesi (hızı buradan alınır)

    def ileri_nokta(self, pos):
        """Drone konumu -> havuç: rotada, drone'un izdüşümünden lookahead metre ileride (x, y, z, v)."""
        w = self.wps                                         # kısaltma
        if len(w) == 1:                                      # tek noktalı rota
            return w[0]                                      # havuç o nokta
        en_j, en_t, en_d = self.i, 0.0, math.inf             # şimdiye kadarki en yakın: parça, oran, uzaklık
        yol = 0.0                                            # drone'un parçasından sonra ne kadar ileri baktık (m)
        j = self.i                                           # aramaya drone'un parçasından başla (geri dönme)
        while j < len(w) - 1 and yol <= ARAMA * self.lookahead:   # parça sayısına değil mesafeye göre sınır
            t = izdusum(pos, w[j], w[j + 1])                 # bu parçadaki gölge
            d = math.dist(pos[:3], ara_nokta(w[j], w[j + 1], t)[:3])  # drone'a uzaklığı
            if d < en_d:                                     # daha yakınsa
                en_j, en_t, en_d = j, t, d                   # bunu hatırla
            if j > self.i:                                   # drone'un kendi parçası sayılmaz
                yol += math.dist(w[j][:3], w[j + 1][:3])     # ileri bakılan mesafeye ekle
            j += 1                                           # sonraki parça
        self.i = en_j                                        # drone artık bu parçada
        nokta = ara_nokta(w[en_j], w[en_j + 1], en_t)        # drone'un rotadaki gölgesi
        self.golge = nokta                                   # target() hızı buradan okuyacak
        kalan = max(L_MIN, self.lookahead * min(1.0, nokta[3] / V_MAX))   # havuç mesafesi hızla orantılı: virajda (yavaşken) yakın, köşe kesmez
        j = en_j                                             # yürüdüğümüz parça
        while True:
            d = math.dist(nokta[:3], w[j + 1][:3])           # parçanın sonuna kalan mesafe
            if d >= kalan:                                   # havuç bu parçanın içinde
                return ara_nokta(nokta, w[j + 1], kalan / d) # tam yerini hesapla
            kalan -= d                                       # parçayı bitir, kalanı sonrakine taşı
            j += 1                                           # sonraki parça
            if j >= len(w) - 1:                              # rota bitti
                return w[-1]                                 # havuç = hedef
            nokta = w[j]                                     # yeni parçanın başı

    def target(self, pos):
        """S-3: drone konumu (x, y, z) -> gidilecek nokta (x, y, z, v). None = rota yok."""
        if not self.wps:                                     # rota hiç verilmemiş
            return None                                      # px4_bridge drone'u yerinde tutar
        son = self.wps[-1]                                   # hedef nokta
        if self.done:                                        # zaten vardıysa
            return (son[0], son[1], son[2], 0.0)             # hedefte dur
        havuc = self.ileri_nokta(pos)                        # havucun yeri
        if math.dist(havuc[:3], son[:3]) < 1e-9 and math.dist(pos[:3], son[:3]) <= VARIS:   # havuç hedefte ve drone çok yakın
            self.done = True                                 # vardık
            return (son[0], son[1], son[2], 0.0)             # hedefte dur
        return (havuc[0], havuc[1], havuc[2], max(self.golge[3], V_MIN))   # havuca, gölgenin hızıyla (en az V_MIN)