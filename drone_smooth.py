"""drone_smooth.py: rotayı kısaltma ve yumuşatma (E-1, E-3). ROS'suz, saf Python."""
import math                                   # karekök, aşağı yuvarlama, uzaklık

from drone_dstar_lite import ROWS             # ızgara satır sayısı (metreye çevirirken y'yi ters çevirmek için)

ORNEK = 4                                     # iki kontrol noktası arasında kaç eğri noktası alınır (adet)
INCELT = 6                                    # eğri duvara taşarsa en fazla kaç kez ara nokta eklenir (adet)
V_MAX = 3.0                                   # en yüksek uçuş hızı (m/s)
A_YAN = 1.5                                   # virajda izin verilen yana ivme (m/s²)
A_MAX = 1.0                                   # hızlanma ve yavaşlama sınırı (m/s²)

def line_free(a, b, grid):
    """a'dan b'ye ip hiçbir dolu hücreye değmiyor mu? Köşe kesme yasak."""
    (x0, y0), (x1, y1) = a, b                 # uç hücreler (x, y)
    dx, dy = abs(x1 - x0), abs(y1 - y0)       # yatayda ve dikeyde kaç fayans gidilecek
    sx = 1 if x1 > x0 else -1                 # x yönü: sağa +1, sola -1
    sy = 1 if y1 > y0 else -1                 # y yönü: aşağı +1, yukarı -1
    x, y, ix, iy = x0, y0, 0, 0               # şu anki fayans ve geçilen çizgi sayıları
    if grid[y][x]:                            # başlangıç fayansı doluysa
        return False                          # yol yok
    while ix < dx or iy < dy:                 # b'ye varana kadar
        dik = (1 + 2 * ix) * dy               # sonraki dikey çizgiye varış (ölçek)
        yat = (1 + 2 * iy) * dx               # sonraki yatay çizgiye varış (ölçek)
        if dik == yat:                        # tam köşe
            if grid[y][x + sx] or grid[y + sy][x]:   # iki yan fayanstan biri dolu mu?
                return False                  # köşe kesme yasak
            x += sx; y += sy; ix += 1; iy += 1        # çapraz bir fayans geç
        elif dik < yat:                       # önce dikey çizgi
            x += sx; ix += 1                  # sağa/sola bir fayans geç
        else:                                 # önce yatay çizgi
            y += sy; iy += 1                  # aşağı/yukarı bir fayans geç
        if grid[y][x]:                        # yeni girdiğimiz fayans dolu mu?
            return False                      # ip engele değiyor
    return True                               # hiçbir dolu fayansa değmedi

def shortcut(path, grid):
    """Zikzak hücre rotasını kısaltır: görülebilen en uzak noktaya atlar (E-1, S-3)."""
    if path is None or len(path) < 3:         # rota yok ya da zaten 1-2 nokta
        return path                           # kısaltılacak bir şey yok
    sonuc = [path[0]]                         # başlangıç noktası hep kalır
    i = 0                                     # şu an bulunduğumuz noktanın sırası
    while i < len(path) - 1:                  # sona varana kadar
        j = len(path) - 1                     # en uzaktaki noktadan başla
        while j > i + 1 and not line_free(path[i], path[j], grid):
            j -= 1                            # görünmüyorsa bir yakına gel
        sonuc.append(path[j])                 # görülen en uzak noktayı ekle
        i = j                                 # ve oraya atla
    return sonuc                              # kısaltılmış rota

def bspline_nokta(p0, p1, p2, p3, t):
    """4 kontrol noktası + t (0..1) -> eğrinin o anki noktası (kübik, düzgün B-spline)."""
    b0 = (1 - t) ** 3 / 6                         # p0'ın ağırlığı (t büyüdükçe azalır)
    b1 = (3 * t**3 - 6 * t**2 + 4) / 6            # p1'in ağırlığı
    b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6   # p2'nin ağırlığı
    b3 = t**3 / 6                                 # p3'ün ağırlığı (t büyüdükçe artar)
    return tuple(b0 * a + b1 * b + b2 * c + b3 * d for a, b, c, d in zip(p0, p1, p2, p3))   # her eksende ağırlıklı ortalama

def bspline(kontrol, ornek=ORNEK):
    """Kontrol noktaları -> eğri noktaları. Eğri tam ilk noktada başlar, tam son noktada biter."""
    k = [kontrol[0]] * 2 + list(kontrol) + [kontrol[-1]] * 2   # uçları 3'er kez yaz: eğri uçlara yapışsın
    egri = []                                                    # eğri noktaları
    for i in range(len(k) - 3):                                  # kayan 4'lü pencere: 0-1-2-3, 1-2-3-4, ...
        for s in range(ornek):                                   # her pencereden ORNEK nokta al
            egri.append(bspline_nokta(k[i], k[i + 1], k[i + 2], k[i + 3], s / ornek))   # t = 0, 0.25, 0.5, 0.75
    egri.append(tuple(kontrol[-1]))                              # son nokta: tam hedef
    return egri

def parca_kareye_degiyor(a, b, x, y):
    """a-b doğru parçası (x, y) hücresinin karesine ([x, x+1] x [y, y+1]) değiyor mu? Köşeye değmek de sayılır."""
    t0, t1 = 0.0, 1.0                                            # parçanın kare içinde kalan kısmı: t0..t1
    dx, dy = b[0] - a[0], b[1] - a[1]                            # parçanın yönü
    for p, q in ((-dx, a[0] - x), (dx, x + 1 - a[0]), (-dy, a[1] - y), (dy, y + 1 - a[1])):   # karenin 4 kenarı
        if p == 0:                                               # parça bu kenara paralel
            if q < 0:                                            # ve karenin dışında
                return False                                     # hiç değmez
        else:
            t = q / p                                            # parça bu kenarı hangi t'de kesiyor
            if p < 0:                                            # karenin içine giriyor
                t0 = max(t0, t)                                  # giriş noktasını ileri al
            else:                                                # kareden çıkıyor
                t1 = min(t1, t)                                  # çıkış noktasını geri al
    return t0 <= t1                                              # kare içinde kalan kısım varsa: değiyor


def egri_guvenli(egri, grid):
    """Eğri (hücre koordinatında) hiçbir dolu hücreye ve ızgara dışına değmiyor mu? Kesin kontrol."""
    for a, b in zip(egri, egri[1:]):                             # art arda iki eğri noktası = küçük parça
        for y in range(math.floor(min(a[1], b[1])), math.floor(max(a[1], b[1])) + 1):      # parçanın kutusundaki satırlar
            for x in range(math.floor(min(a[0], b[0])), math.floor(max(a[0], b[0])) + 1):  # ve sütunlar
                if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):   # ızgara dışı
                    return False                                 # güvenli değil
                if grid[y][x] and parca_kareye_degiyor(a, b, x, y):     # dolu hücreye değiyor
                    return False                                 # güvenli değil
    return True                                                  # hiçbir yere değmedi

def ara_ekle(noktalar):
    """Her iki kontrol noktasının ortasına bir nokta ekler (eğri köşeli çizgiye yaklaşsın)."""
    out = [noktalar[0]]                                          # ilk nokta aynen kalır
    for a, b in zip(noktalar, noktalar[1:]):                     # art arda iki nokta
        out.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))       # ortalarına yeni nokta
        out.append(b)                                            # sonra ikincisi
    return out

def smooth_path(cells, grid, res, z=5.0):
    """S-3: kısaltılmış hücre rotası -> yumuşak eğri [(x, y, z)] metre (map çerçevesi). Güvenlik payına taşmaz."""
    if not cells:                                                # rota yoksa
        return []                                                # boş liste
    kontrol = [(x + 0.5, y + 0.5) for x, y in cells]             # hücre merkezleri (hücre birimi)
    egri = None                                                  # henüz güvenli eğri yok
    for _ in range(INCELT):                                      # en fazla INCELT deneme
        aday = bspline(kontrol)                                  # eğriyi çiz
        if egri_guvenli(aday, grid):                             # duvara değmiyorsa
            egri = aday                                          # bunu al
            break                                                # aramayı bitir
        kontrol = ara_ekle(kontrol)                              # değiyorsa: mıknatısları sıklaştır, tekrar dene
    if egri is None:                                             # hiç güvenli eğri bulunamadıysa
        egri = kontrol                                           # köşeli çizginin kendisi (zaten güvenli)
    return [(px * res, (ROWS - py) * res, z) for px, py in egri]  # metreye çevir (y ters: haritada yukarı)

def viraj_hizi(a, b, c):
    """a-b-c noktalarından geçen dairenin yarıçapına göre b'deki en yüksek güvenli hız (m/s)."""
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)   # üçgenin kenarları (metre)
    s = (ab + bc + ca) / 2                                           # yarım çevre
    alan = math.sqrt(max(s * (s - ab) * (s - bc) * (s - ca), 0.0))   # Heron formülü: üçgenin alanı
    if alan < 1e-12:                                                 # alan yok: noktalar düz çizgide
        return V_MAX                                                 # düzde tam hız
    r = ab * bc * ca / (4 * alan)                                    # 3 noktadan geçen dairenin yarıçapı
    return min(V_MAX, math.sqrt(A_YAN * r))                          # v²/r <= A_YAN  ->  v = √(A_YAN·r)

def speed_profile(points):
    """S-3: [(x, y, z)] metre -> [(x, y, z, v)] uçuş rotası. Virajda yavaş, ilk/son 0, ivme sınırlı."""
    n = len(points)                                                  # nokta sayısı
    if n == 0:                                                       # rota yoksa
        return []                                                    # boş liste
    v = [V_MAX] * n                                                  # başta her yerde tam hız varsay
    for i in range(1, n - 1):                                        # aradaki her nokta için
        v[i] = viraj_hizi(points[i - 1], points[i], points[i + 1])   # viraj ne kadar keskinse o kadar yavaş
    v[0] = v[-1] = 0.0                                               # kalkışta ve varışta dur
    for i in range(1, n):                                            # baştan sona: hızlanma sınırı
        d = math.dist(points[i - 1], points[i])                      # iki nokta arası (metre)
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * A_MAX * d))   # v² = v0² + 2ad'den hızlı olamaz
    for i in range(n - 2, -1, -1):                                   # sondan başa: yavaşlama sınırı
        d = math.dist(points[i], points[i + 1])                      # iki nokta arası (metre)
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * A_MAX * d))   # zamanında fren yap
    return [(x, y, z, vi) for (x, y, z), vi in zip(points, v)]       # S-3 biçimi: (x, y, z, v)

