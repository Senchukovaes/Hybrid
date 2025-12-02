import argparse, os, json, math, hashlib, struct, time

from PIL import Image
import numpy as np
from Crypto.Cipher import AES
import matplotlib.pyplot as plt


# ============================================================
#   Импорт метрик из предыдущих лабораторных работ
# ============================================================

def compute_psnr_ssim(cover_path, stego_path):
    # PSNR и SSIM из Stego_LSB
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.io import imread

    orig = imread(cover_path)
    steg = imread(stego_path)
    psnr = float(peak_signal_noise_ratio(orig, steg, data_range=255))

    h, w = orig.shape[0], orig.shape[1]
    min_side = min(h, w)
    if min_side < 7:
        win_size = min_side if (min_side % 2 == 1) else (min_side - 1)
        if win_size < 3:
            ssim = 1.0 if np.array_equal(orig, steg) else 0.0
            return psnr, float(ssim)
    else:
        win_size = 7

    ssim = structural_similarity(orig, steg, data_range=255, channel_axis=-1, win_size=win_size)
    return psnr, float(ssim)


def save_diff_map(cover_path, stego_path, outpath):
    # Карта разности из Stego_LSB
    orig = np.array(Image.open(cover_path).convert("RGB"), dtype=np.int16)
    steg = np.array(Image.open(stego_path).convert("RGB"), dtype=np.int16)
    diff = np.abs(orig - steg).astype(np.uint8)
    gray_diff = np.max(diff, axis=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_diff, cmap='gray')
    plt.title("Difference map (max channel abs diff)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_histograms(cover_path, stego_path, outdir, base):
    # Гистограммы из Stego_LSB
    os.makedirs(outdir, exist_ok=True)
    orig = np.array(Image.open(cover_path).convert("RGB"), dtype=np.uint8)
    steg = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)

    channels = ['R', 'G', 'B']

    for i, ch in enumerate(channels):
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        plt.suptitle(f"Histogram channel {ch}")

        axes[0].hist(orig[:, :, i].ravel(), bins=256)
        axes[0].set_title("Cover")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        axes[1].hist(steg[:, :, i].ravel(), bins=256)
        axes[1].set_title("Stego")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(outdir, f"{base}_hist_{ch}.png"), dpi=150)
        plt.close()


def chi_square_stat(image_path: str):
    # Хи-квадрат тест
    img = Image.open(image_path).convert("RGB")
    rgb_bytes = img.tobytes()

    def _channel_histogram(rgb_bytes: bytes, channel: int) -> list:
        n_pixels = len(rgb_bytes) // 3
        hist = [0] * 256
        base = channel
        for i in range(n_pixels):
            v = rgb_bytes[3 * i + base]
            hist[v] += 1
        return hist

    def hi2_lsb_channel(rgb_bytes: bytes, channel: int):
        hist = _channel_histogram(rgb_bytes, channel)
        chi2 = 0.0
        used_pairs = 0

        for k in range(0, 256, 2):
            o0 = hist[k]
            o1 = hist[k + 1]
            s = o0 + o1

            if s == 0:
                continue

            e = s / 2.0
            chi2 += (o0 - e) * (o0 - e) / e + (o1 - e) * (o1 - e) / e
            used_pairs += 1

        df = max(used_pairs - 1, 1)
        return chi2, df

    import scipy.stats as stats
    def calculate_p_value(chi2_stat: float, df: int) -> float:
        return 1 - stats.chi2.cdf(chi2_stat, df)

    result = {}
    for ch, name in enumerate(("R", "G", "B")):
        chi2, df = hi2_lsb_channel(rgb_bytes, ch)
        p_value = calculate_p_value(chi2, df)
        result[name] = {
            "chi2_total": float(chi2),
            "df": int(df),
            "p_value": float(p_value)
        }

    total_chi2 = np.mean([ch["chi2_total"] for ch in result.values()])
    total_df = np.mean([ch["df"] for ch in result.values()])
    total_p_value = calculate_p_value(total_chi2, int(total_df))

    return {
        "channels": result,
        "overall": {
            "chi2_total": float(total_chi2),
            "df": int(total_df),
            "p_value": float(total_p_value)
        }
    }


def compute_basic_metrics_crypto(original_path: str, encrypted_path: str) -> dict:
    #Базовые метрики
    orig_img = Image.open(original_path).convert('RGB')
    enc_img = Image.open(encrypted_path).convert('RGB')

    def _image_to_channel_lists(img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        px = img.load()
        R = [[0] * w for _ in range(h)]
        G = [[0] * w for _ in range(h)]
        B = [[0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                R[y][x] = int(r)
                G[y][x] = int(g)
                B[y][x] = int(b)
        return R, G, B

    def entropy_channel(channel):
        flat = []
        for row in channel:
            flat.extend(row)
        total = len(flat)
        if total == 0:
            return 0.0
        hist = [0] * 256
        for v in flat:
            hist[v] += 1
        ent = 0.0
        for count in hist:
            if count == 0:
                continue
            p = count / total
            ent -= p * math.log2(p)
        return float(ent)

    R_o, G_o, B_o = _image_to_channel_lists(orig_img)
    R_e, G_e, B_e = _image_to_channel_lists(enc_img)

    ent_orig = [entropy_channel(R_o), entropy_channel(G_o), entropy_channel(B_o)]
    ent_enc = [entropy_channel(R_e), entropy_channel(G_e), entropy_channel(B_e)]

    return {
        "entropy_original": ent_orig,
        "entropy_processed": ent_enc,
    }


def analyze_pair_metrics(cover_path, stego_path, outdir):
    # Полный анализ метрик для пары изображений
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(stego_path))[0]

    # PSNR/SSIM
    psnr, ssim = compute_psnr_ssim(cover_path, stego_path)

    # Карта разности и гистограммы
    diff_path = os.path.join(outdir, f"{base}_diff.png")
    save_diff_map(cover_path, stego_path, diff_path)
    save_histograms(cover_path, stego_path, outdir, base)

    # Хи-квадрат тест
    chi2res = chi_square_stat(stego_path)

    # Энтропия
    entropy_res = compute_basic_metrics_crypto(cover_path, stego_path)

    result = {
        "cover": os.path.abspath(cover_path),
        "stego": os.path.abspath(stego_path),
        "psnr": psnr,
        "ssim": ssim,
        "chi2": chi2res,
        "entropy": entropy_res,
    }

    # Сохранение результатов
    json_path = os.path.join(outdir, f"{base}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Создание таблицы метрик в PNG
    table_path = os.path.join(outdir, f"{base}_metrics_table.png")
    save_metrics_table(result, table_path)

    return result


def save_metrics_table(metrics_result: dict, out_path: str):
    # metrics_result: словарь с результатами метрик от analyze_pair_metrics
    import matplotlib.pyplot as plt

    # Получаем базовое имя файла для заголовка
    cover_name = os.path.basename(metrics_result["cover"])
    stego_name = os.path.basename(metrics_result["stego"])

    # Создаем фигуру с правильным размером
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')

    # Заголовок
    title = f"Метрики качества стеганографии\n" \
            f"Cover: {cover_name} | Stego: {stego_name}"
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Основные метрики
    cell_text = []
    cell_text.append(["PSNR", f"{metrics_result['psnr']:.2f} dB"])
    cell_text.append(["SSIM", f"{metrics_result['ssim']:.6f}"])

    # Хи-квадрат тест
    chi2 = metrics_result['chi2']['overall']
    cell_text.append(["Хи-квадрат p-value", f"{chi2['p_value']:.6f}"])
    cell_text.append(["Хи-квадрат статистика", f"{chi2['chi2_total']:.2f}"])
    cell_text.append(["Степени свободы", f"{chi2['df']}"])

    # Энтропия
    entropy_orig = metrics_result['entropy']['entropy_original']
    entropy_stego = metrics_result['entropy']['entropy_processed']
    cell_text.append(["Энтропия R (cover/stego)", f"{entropy_orig[0]:.4f} / {entropy_stego[0]:.4f}"])
    cell_text.append(["Энтропия G (cover/stego)", f"{entropy_orig[1]:.4f} / {entropy_stego[1]:.4f}"])
    cell_text.append(["Энтропия B (cover/stego)", f"{entropy_orig[2]:.4f} / {entropy_stego[2]:.4f}"])

    # Цвета для строк (зебра)
    colors = ['#f5f5f5', 'white']
    row_colors = [colors[i % 2] for i in range(len(cell_text))]

    # Создаем таблицу
    table = plt.table(cellText=cell_text,
                      colLabels=["Метрика", "Значение"],
                      colWidths=[0.5, 0.5],
                      cellLoc='left',
                      loc='center',
                      bbox=[0, 0, 1, 0.9])

    # Настраиваем стиль таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Заголовки колонок
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_height(0.08)

    # Строки с данными
    for i in range(1, len(cell_text) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor(row_colors[i - 1])
            table[(i, j)].set_height(0.06)

    # Сохраняем с высоким DPI
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Таблица метрик сохранена: {out_path}")


def save_comparison_table(hybrid_metrics: dict, simple_metrics: dict, out_path: str):
    import matplotlib.pyplot as plt

    cover_name = os.path.basename(hybrid_metrics["cover"])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')

    # Заголовок
    title = f"Сравнение методов стеганографии\nCover: {cover_name}"
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.97)

    # Данные для таблицы
    cell_text = []

    # Основные метрики
    cell_text.append(["PSNR (dB)",
                      f"{hybrid_metrics['psnr']:.2f}",
                      f"{simple_metrics['psnr']:.2f}"])

    cell_text.append(["SSIM",
                      f"{hybrid_metrics['ssim']:.6f}",
                      f"{simple_metrics['ssim']:.6f}"])

    # Хи-квадрат
    cell_text.append(["Хи-квадрат p-value",
                      f"{hybrid_metrics['chi2']['overall']['p_value']:.6f}",
                      f"{simple_metrics['chi2']['overall']['p_value']:.6f}"])

    cell_text.append(["Хи-квадрат статистика",
                      f"{hybrid_metrics['chi2']['overall']['chi2_total']:.2f}",
                      f"{simple_metrics['chi2']['overall']['chi2_total']:.2f}"])

    # Энтропия (средняя по каналам)
    hybrid_entropy_avg = np.mean(hybrid_metrics['entropy']['entropy_processed'])
    simple_entropy_avg = np.mean(simple_metrics['entropy']['entropy_processed'])
    cell_text.append(["Средняя энтропия (stego)",
                      f"{hybrid_entropy_avg:.4f}",
                      f"{simple_entropy_avg:.4f}"])

    # Энтропия по каналам
    for i, channel in enumerate(['R', 'G', 'B']):
        hybrid_ent = hybrid_metrics['entropy']['entropy_processed'][i]
        simple_ent = simple_metrics['entropy']['entropy_processed'][i]
        cell_text.append([f"Энтропия {channel} (stego)",
                          f"{hybrid_ent:.4f}",
                          f"{simple_ent:.4f}"])

    # Определяем лучший метод для каждой метрики
    best_methods = []
    for i, row in enumerate(cell_text):
        try:
            h_val = float(row[1])
            s_val = float(row[2])

            if i in [0, 1, 4]:  # Метрики где выше лучше (PSNR, SSIM, p-value)
                if h_val > s_val:
                    best_methods.append("Гибрид")
                elif s_val > h_val:
                    best_methods.append("Простой")
                else:
                    best_methods.append("Равно")
            elif i in [2, 3, 5]:  # Метрики где меньше лучше (измененные пиксели, chi2 статистика)
                if h_val < s_val:
                    best_methods.append("Гибрид")
                elif s_val < h_val:
                    best_methods.append("Простой")
                else:
                    best_methods.append("Равно")
            else:  # Энтропия - чем ближе значения, тем лучше
                diff = abs(h_val - s_val)
                if diff < 0.001:
                    best_methods.append("Равно")
                elif diff < 0.01:  # Небольшая разница
                    best_methods.append("≈")
                else:
                    best_methods.append("")  # Не определяем лучшего для энтропии
        except:
            best_methods.append("")

    # Добавляем столбец с лучшим методом
    for i, (row, best) in enumerate(zip(cell_text, best_methods)):
        row.append(best)

    # Создаем таблицу
    table = plt.table(cellText=cell_text,
                      colLabels=["Метрика", "Гибридный", "Простой LSB", "Лучший"],
                      colWidths=[0.3, 0.2, 0.2, 0.15],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 0.85])

    # Настраиваем стиль
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Заголовки колонок
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_height(0.07)

    # Цвета строк
    colors = ['#f8f9fa', 'white']
    for i in range(1, len(cell_text) + 1):
        row_color = colors[i % 2]
        for j in range(4):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_height(0.06)

    # Подсчет лучших методов (только для первых 6 метрик, где есть четкое определение)
    valid_metrics = best_methods[:6]
    hybrid_wins = sum(1 for b in valid_metrics if b == "Гибрид")
    simple_wins = sum(1 for b in valid_metrics if b == "Простой")
    equal_wins = sum(1 for b in valid_metrics if b == "Равно")

    summary_text = f"Итог сравнения (6 основных метрик): Гибридный - {hybrid_wins} | Простой - {simple_wins} | Равно - {equal_wins}"
    plt.figtext(0.5, 0.02, summary_text, fontsize=12, weight='bold',
                ha='center', color='#2c3e50')

    # Информация о тесте
    info_text = f"Cover: {cover_name}\n" \
                f"Размер изображения: {hybrid_metrics['total_pixels']:,} пикселей\n" \
                f"Дата анализа: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    plt.figtext(0.02, 0.02, info_text, fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Таблица сравнения сохранена: {out_path}")


# ============================================================
#   простые вспомогательные классы
# ============================================================

class LCGPRNG:

    def __init__(self, seed: int):
        self.m = 2 ** 63
        self.a = 2806196910506780709
        self.c = 1013904223
        self.state = seed & 0xFFFFFFFFFFFFFFFF
        if self.state == 0:
            self.state = 0x12345678

    def next32(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        # Циклический сдвиг битов
        self.state = ((self.state << 13) | (self.state >> 51)) & 0xFFFFFFFFFFFFFFFF
        return self.state & 0xFFFFFFFF

    def rand_byte(self) -> int:
        return self.next32() & 0xFF


class SimpleIVGenerator:

    def __init__(self):
        seed = (int(time.time_ns()) ^ os.getpid()) & 0xFFFFFFFF
        self.state = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2 ** 32

    def next_byte(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state & 0xFF

    def generate(self, n):
        return bytes(self.next_byte() for _ in range(n))


def derive_key_iterative(passphrase: str, salt: bytes, iterations=20000, key_len=16):
    # KDF (генерируем ключ из пароля и соли)
    # Преобразуем пароль в байты, берём хеш и снова байты
    h = hashlib.sha256(passphrase.encode() + salt).digest()
    for _ in range(iterations - 1):
        h = hashlib.sha256(h).digest()
    return h[:key_len]


# ============================================================
#   формат BLOB: фиксированная длина заголовка + ciphertext
# ============================================================

MAGIC = b'HSDG' # идентификатор формата файла
VERSION = 1
SALT_LEN = 16
IV_LEN = 16
HEADER_LEN = 4 + 1 + SALT_LEN + IV_LEN + 4  # 4+1+16+16+4 = 41 байт


def build_blob(salt: bytes, iv: bytes, ct: bytes) -> bytes:
    # HEADER (fixed size) + ciphertext
    header = (
            MAGIC +
            bytes([VERSION]) +
            salt +
            iv +
            struct.pack(">I", len(ct))
    )
    return header + ct


def parse_header(header: bytes):
    #Парсим только HEADER (он не зашифрован)
    if len(header) < HEADER_LEN:
        raise ValueError("HEADER too small")

    if header[:4] != MAGIC:
        raise ValueError("Bad magic")

    version = header[4]
    pos = 5

    salt = header[pos:pos + SALT_LEN]
    pos += SALT_LEN

    iv = header[pos:pos + IV_LEN]
    pos += IV_LEN

    ct_len = struct.unpack(">I", header[pos:pos + 4])[0]

    return salt, iv, ct_len


# ============================================================
#   Image utils
# ============================================================

def image_to_flat(img_path: str):
    # Возвращает плоский массив байтов изображения и его оригинальную форму
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr.flatten(), arr.shape

def get_image_capacity(img_path: str):
    # Возвращает емкость изображения в БИТАХ (LSB на каждый байт)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    # Каждый пиксель имеет 3 канала (RGB), каждый канал - 1 байт, каждый байт дает 1 бит LSB
    return w * h * 3


def flat_to_image(flat, shape, out_path):
    arr = flat.reshape(shape)
    Image.fromarray(arr, "RGB").save(out_path)


def bits_from_bytes(b: bytes):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bytes_from_bits(bits: np.ndarray):
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    return np.packbits(bits).tobytes()


# ============================================================
#   Псевдослучайные позиции (для гибридного метода)
# ============================================================

def make_prng_positions(total_slots: int, n_bits: int, password: str):
    # Получаем детерминированный seed = SHA256(password)[0:8]
    h = hashlib.sha256(password.encode()).digest()
    seed = int.from_bytes(h[:8], "big")

    # генерируем псевдослучайные числа
    pr = LCGPRNG(seed)

    idxs = list(range(total_slots))
    # Перемешиваем по Фишеру-Йейтсу
    for i in range(total_slots - 1, 0, -1):
        j = pr.next32() % (i + 1)
        idxs[i], idxs[j] = idxs[j], idxs[i]
    return idxs[:n_bits]


# ============================================================
#   AES encrypt/decrypt
# ============================================================

def encrypt_message_bytes(plain: bytes, password: str, kdf_iters=20000):
    salt = SimpleIVGenerator().generate(SALT_LEN)
    iv = SimpleIVGenerator().generate(IV_LEN)
    key = derive_key_iterative(password, salt, iterations=kdf_iters)

    cipher = AES.new(key, AES.MODE_CBC, iv)

    # PKCS#7 padding
    pad_len = 16 - (len(plain) % 16)
    padded = plain + bytes([pad_len] * pad_len)

    ct = cipher.encrypt(padded)
    #print (build_blob(salt, iv, ct))
    return build_blob(salt, iv, ct)


def decrypt_blob(blob: bytes, password: str, kdf_iters=20000):
    header = blob[:HEADER_LEN]
    salt, iv, ct_len = parse_header(header)

    if len(blob) < HEADER_LEN + ct_len:
        raise ValueError("Blob слишком короткий для указанной длины ciphertext")

    ct = blob[HEADER_LEN:HEADER_LEN + ct_len]

    key = derive_key_iterative(password, salt, iterations=kdf_iters)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = cipher.decrypt(ct)

    # PKCS#7 unpadding
    pad_len = padded[-1]
    if pad_len > 16 or pad_len <= 0:
        raise ValueError("Invalid padding")

    # Проверяем padding
    for i in range(1, pad_len + 1):
        if padded[-i] != pad_len:
            raise ValueError("Invalid padding bytes")

    return padded[:-pad_len]


# ============================================================
#   EMBED - Гибридный метод (ИСПРАВЛЕННЫЙ)
# ============================================================

def embed_blob(cover_path, out_path, blob: bytes, password: str, payload=None):
    # Встраивание blob с учетом ограничения payload
    flat, shape = image_to_flat(cover_path)
    total_bytes = flat.size  # capacity in bytes (каждый байт дает 1 бит LSB)

    bits = bits_from_bytes(blob)
    n_bits = len(bits)

    # Если payload задан, проверяем, влезает ли blob
    if payload is not None:
        max_bits = int(total_bytes * payload)  # total_bytes, потому что каждый байт дает 1 бит LSB
        if n_bits > max_bits:
            raise ValueError(f"Blob слишком велик для указанного payload: {n_bits} > {max_bits} bits")

    if n_bits > total_bytes:
        raise ValueError(f"Недостаточно места для встраивания {n_bits} битов (capacity {total_bytes} bits).")

    # генерируем псевдослучайные позиции
    positions = make_prng_positions(total_bytes, n_bits, password)

    flat2 = flat.copy()
    mask = np.uint8(0xFE) #1111110
    for i, pos in enumerate(positions):
        bit = int(bits[i]) # бит из сообщения
        flat2[pos] = (flat2[pos] & mask) | bit

    # сохраняем стегоизображение
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    flat_to_image(flat2, shape, out_path)

    meta = {
        "method": "hybrid",
        "len_bits": n_bits,
        "payload_fraction": payload,
        "blob_size_bytes": len(blob),
        "capacity_bytes": total_bytes,
        "capacity_bits": total_bytes,  # каждый байт дает 1 бит LSB
        "cover": os.path.abspath(cover_path),
        "stego": os.path.abspath(out_path)
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ============================================================
#   EMBED - Простой LSB-1 метод (ИСПРАВЛЕННЫЙ)
# ============================================================

def embed_simple_lsb(cover_path, out_path, message_bytes: bytes, payload=None):
    # Простой LSB-1 метод с явной длиной сообщения в первых 4 байтах
    flat, shape = image_to_flat(cover_path)
    total_bytes = flat.size  # capacity in bytes (каждый байт дает 1 бит LSB)

    # Добавляем длину сообщения в первых 4 байтах
    msg_len = len(message_bytes)
    len_bytes = struct.pack(">I", msg_len)  # 4 байта
    final_bytes = len_bytes + message_bytes

    bits = bits_from_bytes(final_bytes)
    n_bits = len(bits)

    # Проверяем capacity
    if payload is not None:
        max_bits = int(total_bytes * payload)
        if n_bits > max_bits:
            raise ValueError(f"Сообщение слишком велико для указанного payload: {n_bits} > {max_bits} bits")

    if n_bits > total_bytes:
        raise ValueError(f"Недостаточно места для встраивания {n_bits} битов (capacity {total_bytes} bits).")

    flat2 = flat.copy()
    mask = np.uint8(0xFE)
    bits_arr = bits.astype(np.uint8)

    # последовательное встраивание
    flat2[:n_bits] = (flat2[:n_bits] & mask) | bits_arr

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    flat_to_image(flat2, shape, out_path)

    meta = {
        "method": "simple_lsb",
        "len_bits": n_bits,
        "payload_fraction": payload,
        "message_len": msg_len,
        "capacity_bytes": total_bytes,
        "capacity_bits": total_bytes,  # каждый байт дает 1 бит LSB
        "cover": os.path.abspath(cover_path),
        "stego": os.path.abspath(out_path)
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ============================================================
#   EXTRACT - Гибридный метод (ИСПРАВЛЕННЫЙ)
# ============================================================

def extract_blob(stego_path, password: str):
    # Извлечение blob по тем же псевдослучайным позициям
    flat, shape = image_to_flat(stego_path)
    total = flat.size

    # 1) Извлекаем HEADER_LEN*8 бит для получения заголовка
    header_bits_count = HEADER_LEN * 8
    header_positions = make_prng_positions(total, header_bits_count, password)

    header_bits = np.array([flat[p] & 1 for p in header_positions], dtype=np.uint8)
    header_bytes = bytes_from_bits(header_bits)

    salt, iv, ct_len = parse_header(header_bytes)

    # 2) Вычисляем общее количество бит для всего blob
    total_blob_len = HEADER_LEN + ct_len
    total_bits_needed = total_blob_len * 8

    # 3) Получаем все позиции для всего blob
    positions = make_prng_positions(total, total_bits_needed, password)

    # 4) Извлекаем все биты
    all_bits = np.array([flat[p] & 1 for p in positions], dtype=np.uint8)
    blob = bytes_from_bits(all_bits)

    return blob


# ============================================================
#   EXTRACT - Простой LSB-1 метод (ИСПРАВЛЕННЫЙ)
# ============================================================

def extract_simple_lsb(stego_path):
    # Извлечение сообщения с явной длиной в первых 4 байтах
    flat, shape = image_to_flat(stego_path)

    # Сначала извлекаем 32 бита (4 байта) для длины
    len_bits = (flat[:32] & 1).astype(np.uint8)
    len_bytes = np.packbits(len_bits).tobytes()

    if len(len_bytes) < 4:
        raise ValueError("Не удалось извлечь длину сообщения")

    msg_len = struct.unpack(">I", len_bytes[:4])[0]

    # Вычисляем общее количество бит для извлечения
    total_bits_needed = 32 + msg_len * 8  # 4 байта длины + само сообщение

    if total_bits_needed > len(flat):
        raise ValueError(f"Требуется {total_bits_needed} бит, но доступно только {len(flat)}")

    # Извлекаем все биты сообщения
    all_bits = (flat[:total_bits_needed] & 1).astype(np.uint8)
    all_bytes = np.packbits(all_bits).tobytes()

    # Пропускаем первые 4 байта (длину)
    msg_bytes = all_bytes[4:4 + msg_len]

    # Корректная декодировка
    def safe_decode(data: bytes) -> str:
        for enc in ("utf-8", "cp1251", "latin-1", "utf-8-sig"):
            try:
                return data.decode(enc)
            except:
                pass
        return data.decode("latin-1", errors="replace")

    return safe_decode(msg_bytes), msg_bytes


# ============================================================
#   CLI (ИСПРАВЛЕННЫЙ)
# ============================================================

def cmd_embed(args):
    with open(args.msg, "rb") as f:
        plain = f.read()

    #print(f"Размер исходного сообщения: {len(plain)} байт")

    # Получаем емкость изображения
    img = Image.open(args.cover).convert("RGB")
    w, h = img.size
    capacity_bytes = w * h * 3  # каждый пиксель RGB = 3 байта
    #print(f"Емкость изображения: {capacity_bytes} байт ({capacity_bytes} бит LSB)")

    if args.simple:
        # Простой LSB-1 метод
        if args.password:
            print("[!] Пароль игнорируется для простого LSB метода")

        # Рассчитываем максимальный размер сообщения
        if args.payload is not None:
            max_bits = int(capacity_bytes * args.payload)
            # Нужно место для 4 байт длины
            max_msg_bits = max_bits - 32
            if max_msg_bits < 0:
                print(f"[ОШИБКА] Payload слишком мал для хранения даже длины сообщения")
                return

            max_msg_bytes = max_msg_bits // 8
            if len(plain) > max_msg_bytes:
                #print(f"[!] Сообщение обрезано с {len(plain)} до {max_msg_bytes} байт")
                plain = plain[:max_msg_bytes]

        try:
            meta = embed_simple_lsb(args.cover, args.out, plain, args.payload)
            #print(f"Встроено простым LSB методом. Встроено байт: {meta['message_len']}")
            #print(f"Использовано бит LSB: {meta['len_bits']}")
        except ValueError as e:
            print(f"[ОШИБКА] {e}")
            print("Попробуйте уменьшить payload или размер сообщения")
            return

    else:
        # Гибридный метод
        if not args.password:
            print("[ОШИБКА] Для гибридного метода требуется пароль")
            return

        # Рассчитываем максимальный размер сообщения с учетом payload
        if args.payload is not None:
            max_bits = int(capacity_bytes * args.payload)

            # Место для заголовка (41 байт = 328 бит)
            header_bits = HEADER_LEN * 8
            remaining_bits = max_bits - header_bits

            if remaining_bits < 128:  # Минимум 16 байт для AES
                print(f"[ОШИБКА] Payload слишком мал. Нужно хотя бы {header_bits + 128} бит")
                return

            # AES добавляет padding до 16 байт
            # Максимальный размер ciphertext, который можно встроить
            max_ct_bytes = remaining_bits // 8

            # Шифруем тестовое сообщение, чтобы узнать размер ciphertext
            test_salt = SimpleIVGenerator().generate(SALT_LEN)
            test_iv = SimpleIVGenerator().generate(IV_LEN)
            test_key = derive_key_iterative(args.password, test_salt, iterations=args.kdf_iters)

            cipher = AES.new(test_key, AES.MODE_CBC, test_iv)
            pad_len = 16 - (len(plain) % 16)
            padded_len = len(plain) + pad_len
            ct_len = padded_len  # После шифрования размер будет таким же

            # Полный размер blob
            blob_len = HEADER_LEN + ct_len
            blob_bits = blob_len * 8

            if blob_bits > max_bits:
                # Нужно обрезать сообщение
                # Оцениваем максимальный размер plaintext
                # blob_bits = HEADER_LEN*8 + (plain_len + pad_len)*8 ≤ max_bits
                # plain_len ≤ (max_bits/8) - HEADER_LEN - pad_len
                max_plain_bytes = (max_bits // 8) - HEADER_LEN - 16  # Консервативная оценка

                if max_plain_bytes <= 0:
                    print(f"[ОШИБКА] Payload слишком мал для любого сообщения")
                    return

                if len(plain) > max_plain_bytes:
                    #print(f"[!] Сообщение обрезано с {len(plain)} до {max_plain_bytes} байт")
                    plain = plain[:max_plain_bytes]

        # Шифруем и встраиваем
        blob = encrypt_message_bytes(plain, args.password, args.kdf_iters)
        #print(f"Размер blob (с заголовком): {len(blob)} байт ({len(blob) * 8} бит)")

        try:
            meta = embed_blob(args.cover, args.out, blob, args.password, args.payload)
            #print(f"Встроено гибридным методом. Встроено бит: {meta['len_bits']}")
            #print(f"Использовано {meta['len_bits'] / meta['capacity_bits'] * 100:.2f}% емкости")
        except ValueError as e:
            print(f"[ОШИБКА] {e}")
            return

    # Расчет метрик
    if args.metrics:
        #print("Вычисление метрик...")
        results_dir = os.path.join(os.path.dirname(args.out), "metrics")
        metrics_result = analyze_pair_metrics(args.cover, args.out, results_dir)
        print(f"Метрики сохранены в {results_dir}")


def cmd_extract(args):
    if args.simple:
        # Простой LSB-1 метод
        if args.password:
            print("[!] Пароль игнорируется для простого LSB метода")

        try:
            msg_str, msg_bytes = extract_simple_lsb(args.stego)
            with open(args.out, "wb") as f:
                f.write(msg_bytes)
            #print(f"Извлечено простым LSB методом. Сообщение: {len(msg_bytes)} байт")
            print("Превью сообщения:", (msg_str[:200] + "...") if len(msg_str) > 200 else msg_str)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось извлечь сообщение: {e}")

    else:
        # Гибридный метод
        if not args.password:
            print("[ОШИБКА] Для гибридного метода требуется пароль")
            return

        try:
            blob = extract_blob(args.stego, args.password)
            #print(f"Извлечен blob размером: {len(blob)} байт")

            plain = decrypt_blob(blob, args.password, args.kdf_iters)
            with open(args.out, "wb") as f:
                f.write(plain)
            #print(f"Извлечено гибридным методом. Сообщение: {len(plain)} байт")

            # Показываем превью
            try:
                preview = plain.decode('utf-8')[:200]
                print("Превью сообщения:", preview + ("..." if len(plain) > 200 else ""))
            except:
                print("Сообщение содержит бинарные данные, предпросмотр недоступен")

        except Exception as e:
            print(f"[ОШИБКА] Не удалось извлечь сообщение: {e}")
            import traceback
            traceback.print_exc()


def cmd_metrics(args):
    # Отдельная команда для расчета метрик
    results_dir = os.path.join(os.path.dirname(args.stego), "metrics")
    metrics_result = analyze_pair_metrics(args.cover, args.stego, results_dir)

    print("\n=== РЕЗУЛЬТАТЫ МЕТРИК ===")
    print(f"PSNR: {metrics_result['psnr']:.2f} dB")
    print(f"SSIM: {metrics_result['ssim']:.6f}")
    print(f"Измененные пиксели: {metrics_result['changed_pixels']} ({metrics_result['changed_percent']:.2f}%)")
    print(f"Chi2 p-value: {metrics_result['chi2']['overall']['p_value']:.6f}")
    print(f"Энтропия исходная: {metrics_result['entropy']['entropy_original']}")
    print(f"Энтропия стего: {metrics_result['entropy']['entropy_processed']}")
    print(f"\nПолные результаты сохранены в: {results_dir}")


def cmd_compare(args):
    #Сравнение двух методов на одном изображении
    with open(args.msg, "rb") as f:
        plain = f.read()

    print(f"Размер тестового сообщения: {len(plain)} байт")

    base_name = os.path.splitext(os.path.basename(args.cover))[0]
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)

    # Проверяем capacity
    flat, shape = image_to_flat(args.cover)
    capacity_bits = flat.size

    if args.payload is not None:
        max_bits = int(capacity_bits * args.payload)
        print(f"Максимальное количество бит для payload={args.payload}: {max_bits}")
    else:
        max_bits = capacity_bits

    # Гибридный метод
    hybrid_out = os.path.join(results_dir, f"{base_name}_hybrid.png")
    try:
        blob = encrypt_message_bytes(plain, args.password, args.kdf_iters)
        blob_bits = len(blob) * 8

        if blob_bits > max_bits:
            print(f"[!] Blob гибридного метода ({blob_bits} бит) не влезает в payload")
            print("   Уменьшайте сообщение или увеличивайте payload")
            return

        embed_blob(args.cover, hybrid_out, blob, args.password, args.payload)
        print(f"Гибридный метод: встроено {blob_bits} бит")
    except Exception as e:
        print(f"[ОШИБКА] Гибридный метод: {e}")
        return

    # Простой LSB-1 метод
    simple_out = os.path.join(results_dir, f"{base_name}_simple.png")
    try:
        # Добавляем 4 байта для длины
        total_bytes = len(plain) + 4
        total_bits = total_bytes * 8

        if total_bits > max_bits:
            print(f"[!] Сообщение простого метода ({total_bits} бит) не влезает в payload")
            return

        embed_simple_lsb(args.cover, simple_out, plain, args.payload)
        print(f"Простой метод: встроено {total_bits} бит")
    except Exception as e:
        print(f"[ОШИБКА] Простой метод: {e}")
        return

    # Расчет метрик для обоих методов
    print("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
    print("Гибридный метод:")
    hybrid_metrics = analyze_pair_metrics(args.cover, hybrid_out,
                                          os.path.join(results_dir, "hybrid_metrics"))

    print("\nПростой LSB метод:")
    simple_metrics = analyze_pair_metrics(args.cover, simple_out,
                                          os.path.join(results_dir, "simple_metrics"))

    # Создаем таблицу сравнения
    comparison_table_path = os.path.join(results_dir, f"{base_name}_comparison_table.png")
    save_comparison_table(hybrid_metrics, simple_metrics, comparison_table_path)

    print("\n=== СВОДКА ===")
    print(f"{'Метрика':<20} {'Гибридный':<10} {'Простой':<10} {'Лучше':<10}")
    print("-" * 50)

    metrics_comparison = [
        ("PSNR (dB)", hybrid_metrics["psnr"], simple_metrics["psnr"], "выше"),
        ("SSIM", hybrid_metrics["ssim"], simple_metrics["ssim"], "выше"),
        ("Изменено %", hybrid_metrics["changed_percent"], simple_metrics["changed_percent"], "ниже"),
        ("Chi2 p-value", hybrid_metrics["chi2"]["overall"]["p_value"],
         simple_metrics["chi2"]["overall"]["p_value"], "выше")
    ]

    for name, h_val, s_val, better in metrics_comparison:
        h_better = (h_val > s_val) if better == "выше" else (h_val < s_val)
        better_method = "Гибрид" if h_better else "Простой"
        print(f"{name:<20} {h_val:<10.4f} {s_val:<10.4f} {better_method:<10}")

    print(f"\nРезультаты сохранены в папке: {results_dir}")


def main():
    p = argparse.ArgumentParser(description="Гибридный крипто-стего инструмент")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Embed command
    p1 = sub.add_parser("embed", help="Встраивание сообщения")
    p1.add_argument("--cover", required=True, help="Исходное изображение")
    p1.add_argument("--out", required=True, help="Выходное стего-изображение")
    p1.add_argument("--msg", required=True, help="Файл с сообщением")
    p1.add_argument("--password", help="Пароль для шифрования (требуется для гибридного метода)")
    p1.add_argument("--payload", type=float, default=None, help="Доля полезной нагрузки (0.0-1.0)")
    p1.add_argument("--kdf-iters", type=int, default=20000, help="Количество итераций KDF")
    p1.add_argument("--simple", action="store_true", help="Использовать простой LSB вместо гибридного метода")
    p1.add_argument("--metrics", action="store_true", help="Рассчитать метрики после встраивания")
    p1.set_defaults(func=cmd_embed)

    # Extract command
    p2 = sub.add_parser("extract", help="Извлечение сообщения")
    p2.add_argument("--stego", required=True, help="Стего-изображение")
    p2.add_argument("--out", required=True, help="Файл для восстановленного сообщения")
    p2.add_argument("--password", help="Пароль для расшифровки (требуется для гибридного метода)")
    p2.add_argument("--kdf-iters", type=int, default=20000, help="Количество итераций KDF")
    p2.add_argument("--simple", action="store_true", help="Использовать простой LSB вместо гибридного метода")
    p2.set_defaults(func=cmd_extract)

    # Metrics command
    p3 = sub.add_parser("metrics", help="Расчет метрик для пары изображений")
    p3.add_argument("--cover", required=True, help="Исходное изображение")
    p3.add_argument("--stego", required=True, help="Стего-изображение")
    p3.set_defaults(func=cmd_metrics)

    # Compare command
    p4 = sub.add_parser("compare", help="Сравнение гибридного и простого методов")
    p4.add_argument("--cover", required=True, help="Исходное изображение")
    p4.add_argument("--msg", required=True, help="Файл с сообщением")
    p4.add_argument("--password", required=True, help="Пароль для гибридного метода")
    p4.add_argument("--payload", type=float, default=0.05, help="Доля полезной нагрузки")
    p4.add_argument("--kdf-iters", type=int, default=20000, help="Количество итераций KDF")
    p4.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()