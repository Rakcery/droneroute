"""drone_smooth.py: rotayı kısaltma ve yumuşatma (E-1, E-3). ROS'suz, saf Python."""


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