# Watch & Learn

YouTube videolarını Claude Code'a "izlettiren" bir pipeline. Transcript + ekran görüntülerini birleştirerek bir storyboard grid oluşturur, Claude bunu Vision LLM ile analiz eder ve bilgileri `docs/wiki/` altında Obsidian formatında saklar.

## Nasıl Çalışır

```
YouTube URL
    ↓
Transcript İndir (VTT/SRT → whisper.cpp → Groq Whisper fallback)
    ↓
NLP Analizi → Akıllı Timestamp Seçimi
    ↓
FFmpeg → Sadece O Anların Frame'leri
    ↓
Storyboard Grid (JPEG)
    ↓
Claude (Vision LLM) → Analiz
    ↓
docs/wiki/ → Obsidian Knowledge Base
```

Sadece transcript kullanmak yeterli değil — önemli bilgilerin büyük kısmı ekranda gösterilir, sesle anlatılmaz. Bu pipeline transcript'i NLP ile okuyarak *nerede* ekran görüntüsü alınacağını öğrenir; sabit aralıklı frame extraction yerine semantik olarak değerli anları seçer.

## Kurulum

**Gereksinimler:** Python 3.11+, ffmpeg, Claude Code

### 1. Repoyu klonla

```bash
git clone https://github.com/ozanzeng/watch-learn
cd watch-learn
```

### 2. Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. spaCy dil modelini indir

```bash
python -m spacy download en_core_web_sm
```

### 5. ffmpeg yükle

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
# Windows: https://ffmpeg.org/download.html
```

### 6. (Opsiyonel) Yerel whisper.cpp

YouTube altyazısı olmayan videolar ve yerel dosyalar için önerilen yol budur, ses makineden çıkmaz:

```bash
brew install whisper-cpp
# ggml modelini indir (ör. ggml-medium.bin): https://huggingface.co/ggerganov/whisper.cpp
export WHISPER_CPP_MODEL="/path/to/ggml-medium.bin"
```

### 7. (Opsiyonel) Groq API Key

whisper.cpp yoksa Groq Whisper fallback (ses Groq'a gönderilir):

```bash
export GROQ_API_KEY="gsk_..."
```

[Groq API key al](https://console.groq.com) — ücretsiz tier yeterli.

## Kullanım

### CLI

```bash
source .venv/bin/activate

watch-learn "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output-dir ./output \
  --max-frames 20 \
  --verbose
```

Storyboard JPEG'leri `output/VIDEO_ID/` altına kaydedilir. Aynı klasöre `report.md` (süre, çözünürlük, fps, transkript kaynağı) ve `transcript.txt` (zaman damgalı tam metin) de yazılır.

URL yerine yerel bir video dosyası da verilebilir. Yanında aynı adlı `.srt`/`.vtt` varsa transkript olarak o kullanılır, yoksa ses yazıya dökülür:

```bash
watch-learn ~/Movies/IMG_1234.MOV --interval 1 --lang tr
```

### Kurgu ritmi analizi

```bash
watch-learn "https://www.youtube.com/watch?v=VIDEO_ID" --cuts --keep-video
```

`--cuts` montaj ve kurgu referansları içindir. Şunları üretir:
- `shots_page_*.jpg`: her plandan bir frame; rozette plan no, zaman ve süre. Uzun planlardan ~3 sn'de bir ek frame alınır.
- `report.md` içinde:
  - plan sayısı, dakikadaki kesim sayısı
  - plan süreleri (ortalama, medyan, dağılım)
  - 10/30 sn'lik dilimlerde kesim sayısı
  - ses yüksekliği (LUFS, LRA, tepe)
  - tahmini BPM
  - ses vurgusuna ±70 ms içinde denk gelen kesimlerin oranı, rastgele kesimde beklenen oranla birlikte
  - plan listesi

### Claude Code ile

Claude Code kuruluysa projeyi açın ve bir YouTube URL paylaşın. `CLAUDE.md` sayesinde Claude otomatik olarak `watch-learn` skill'ini devreye alır, videoyu analiz eder ve `docs/wiki/` altına yazar.

```
# Claude Code'a şunu yapıştırın:
https://www.youtube.com/watch?v=VIDEO_ID
```

## CLI Seçenekleri

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--output-dir / -o` | `.` | Storyboard JPEG'lerinin kaydedileceği klasör |
| `--max-frames / -n` | `30` | Maksimum frame sayısı |
| `--jpeg-quality / -q` | `85` | JPEG kalitesi (1–95) |
| `--silence-gap / -g` | `5.0` | Sessizlik eşiği (saniye) |
| `--groq-api-key / -k` | env `GROQ_API_KEY` | Whisper fallback için Groq key |
| `--interval / -i` | — | Transkripte bakmadan her N saniyede bir frame |
| `--cuts` | kapalı | Kurgu ritmi analizi: kesimler, plan başına frame, ses vurguları, LUFS |
| `--cut-threshold` | `3.0` | Bir frame'in komşularına göre kaç kat değişince kesim sayılacağı |
| `--whisper-model` | env `WHISPER_CPP_MODEL` | Altyazısız videolar için yerel whisper.cpp ggml modeli |
| `--lang` | `auto` | whisper.cpp konuşma dili (`tr`, `en`, `auto`) |
| `--keep-video` | kapalı | İndirilen videoyu çıktı klasörüne `video.mp4` olarak kopyala |
| `--keep-temp` | kapalı | Geçici video/frame dosyalarını sakla |
| `--no-learn` | kapalı | Bu çalıştırmada keyword öğrenmesini atla |
| `--verbose / -v` | kapalı | Debug log'larını göster |

## Proje Yapısı

```
watch_learn/
  __init__.py      # Veri modelleri (dataclass'lar)
  downloader.py    # yt-dlp ile transcript + video indirme
  analyzer.py      # NLP timestamp seçimi + self-learning
  editing.py       # Kesim tespiti, plan süreleri, ses vurguları, LUFS
  extractor.py     # ffmpeg frame extraction
  compiler.py      # Pillow storyboard grid
  main.py          # CLI entry point

.claude/skills/
  watch-learn/     # Claude Code skill talimatları
  wiki-schema/     # Wiki yazma kuralları

docs/wiki/
  Index.md         # Ana harita
  Videos.md        # Analiz edilmiş videoların kayıtları
  *.md             # Konu bazlı wiki sayfaları

data/
  keyword_store.json   # Self-learning keyword veritabanı

output/
  VIDEO_ID/
    storyboard_page_001.jpg
    storyboard_page_002.jpg
    shots_page_001.jpg     # --cuts ile
    report.md
    transcript.txt
    ...
```

## Nasıl Çalışıyor: Teknik Detay

### Akıllı Timestamp Seçimi

İki kural birlikte çalışır:

**Kural A — Keyword eşleşmesi:** Transcript'te "look at this", "here we have", "diagram", "terminal" gibi konuşmacının ekrana işaret ettiği ifadeler tespit edilir. spaCy ile fiil + demonstrative kombinasyonları da yakalanır.

**Kural B — Sessizlik tespiti:** Konuşmacının 5+ saniye sessiz kaldığı anlar genellikle demo veya görsel içerik gösteriyor demektir.

`--max-frames` sınırı aşılırsa adaylar videonun tamamına yayılır: zaman çizelgesi eşit dilimlere bölünür, her dilimden en iyi aday kalır. Hiç aday olmayan uzun bölümlere eşit aralıkla ek frame konur.

### Kesim tespiti

Her frame (64x36 RGB, 30 fps) bir öncekiyle karşılaştırılır. Bir frame ±2 komşusundan `--cut-threshold` kat fazla değişiyorsa kesim sayılır. Bu adaptif yaklaşım sayesinde:
- hızlı kamera hareketi kesim sayılmaz
- parlaklığı benzer iki plan arasındaki kesim kaçmaz (ffmpeg'in `scene` skoru ağırlıkla parlaklığa baktığı için bunu yakalayamıyor)

0.25 sn'den sık ardışık tespitler (flaş, efektli geçiş) tek kesim sayılır. Ses vurguları spectral flux ile bulunur, tempo ise vurgu zarfının otokorelasyonundan tahmin edilir; ikisi de yalnızca numpy kullanır.

### Self-Learning

Her video bittikten sonra TF-IDF çalışır: seçilen timestamp'lere yakın metinler ile arka plan metinleri karşılaştırılır, yüksek lift'li terimler `data/keyword_store.json`'a yazılır. Sistem izledikçe domain'e özgü vocabulary öğrenir.

### Storyboard Grid

Frame'ler kırpılmadan, kaynağın en-boy oranıyla yerleştirilir. Dikey videolarda (Shorts, Reels, telefon) sayfa başına 8 frame (4x2, 480px) kullanılır. Yatay videolarda frame sayısına göre adaptif çözünürlük:

| Frame | Grid | Hücre |
|---|---|---|
| 1 | 1×1 | 1280px |
| 2 | 2×1 | 1024px |
| 3–4 | 2×2 | 960px |
| 5–6 | 3×2 | 800px |
| 7–9 | 3×3 | 720px |
| 10–12 | 4×3 | 640px |
| 13+ | 3×3 sayfalar | 720px |

12'den fazla frame otomatik olarak birden fazla JPEG sayfasına bölünür.

## Bağımlılıklar

| Paket | Amaç |
|---|---|
| `yt-dlp` | YouTube transcript + video indirme |
| `Pillow` | Storyboard grid oluşturma |
| `spaCy` | NLP timestamp analizi |
| `scikit-learn` | TF-IDF self-learning |
| `webvtt-py` | VTT altyazı parse |
| `click` | CLI |
| `groq` | Whisper API fallback |
| `numpy` | Kesim ve ses vurgusu analizi |

## Sorun Giderme

| Hata | Çözüm |
|---|---|
| `ffmpeg not found` | `brew install ffmpeg` |
| `No module named spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| `Cannot access video` | Video private, yaş kısıtlı veya bölge engelinde olabilir |
| Transcript bulunamadı | `WHISPER_CPP_MODEL` ile yerel whisper.cpp ya da `--groq-api-key` ile Groq kullan |
| `Module import error` | Venv aktif mi kontrol et: `source .venv/bin/activate && pip install -e .` |
