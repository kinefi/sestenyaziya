# Ses'ten Yazıya

Türkçe ses kayıtlarını gerçek zamanlı olarak metne dönüştüren web uygulaması.

[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) ile transkripsiyon, [Resemblyzer](https://github.com/resemble-ai/resemblyzer) ile konuşmacı ayrıştırma ve [Gradio](https://www.gradio.app/) ile arayüz sağlar.

## Özellikler

- Canlı akış — segmentler tamamlandıkça ekrana yansır
- Model seçimi — small (hızlı), medium (dengeli), large-v3 (kaliteli)
- Konuşmacı ayrıştırma — her konuşmacı ayrı paragraf olarak etiketlenir (token gerektirmez)
- Otomatik konuşmacı sayısı tespiti (1-10 arası) veya manuel giriş
- Otomatik cihaz tespiti (CUDA/GPU veya CPU) ve CUDA hatalarında otomatik CPU (`int8_float32`) fallback.
- Kalıcı önbellekleme (transkripsiyon ve konuşmacı imzaları için)
- SHA-256 tabanlı dosya doğrulama ile mükemmel önbellek eşleşmesi.
- Konuşmacı ayrıştırma sürecinde performans iyileştirmeleri (binary search tabanlı).
- Transkripsiyon tamamlandığında `.txt`, `.srt` ve `.vtt` formatlarında indirilebilir.
- Duraklat / Devam et / Durdur desteği
- Gelişmiş önbellek yönetimi arayüzü (Boyut sınırı ve temizleme)

## Kurulum

```bash
uv sync
```

> İlk çalıştırmada Whisper modeli (~500 MB) ve Resemblyzer encoder'ı (~17 MB) Hugging Face Hub'dan otomatik indirilir.

## Kullanım

```bash
uv run python main.py
```

### Seçenekler

| Parametre | Açıklama | Varsayılan |
| --- | --- | --- |
| `--model` | Whisper model boyutu (small, medium, large-v3) | `medium` |
| `--port` | Web arayüzü portu | `7860` |
| `--paragraph-pause` | Paragraf sonu tespiti için saniye | `1.5` |
| `--share` | Gradio paylaşım linki oluşturur | `False` |
```

### Hot reload (geliştirme)

```bash
uv run gradio main.py
```

## Proje Yapısı

```text
├── app/
│   ├── config.py         # ModelSize enum, sabitler, cihaz tespiti
│   ├── models.py         # Whisper ve VoiceEncoder yükleme, duraklatma/durdurma eventleri
│   ├── diarization.py    # Ses yükleme (PyAV), konuşmacı ayrıştırma (KMeans)
│   ├── transcription.py  # Transkripsiyon generator'ı
│   ├── cache_utils.py    # Önbellek yönetimi, dosya hashing ve temizleme işlemleri
│   └── ui.py             # Gradio arayüzü ve callback'ler
├── main.py               # Giriş noktası
├── pyproject.toml
└── README.md
```

## Bağımlılıklar

| Paket | Görev |
| --- | --- |
| `faster-whisper` | Hızlı Whisper implementasyonu |
| `ctranslate2` | Model hızlandırma ve cihaz yönetimi |
| `resemblyzer` | Konuşmacı gömme vektörleri |
| `scikit-learn` | KMeans kümeleme (konuşmacı ayrıştırma) |
| `av` (PyAV) | Bellek dostu ses akışı ve işleme |
| `torch` (CPU) | Resemblyzer için PyTorch arka ucu |
| `gradio` | Web arayüzü |
