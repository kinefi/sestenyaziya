# Ses'ten Yazıya

Türkçe ses kayıtlarını gerçek zamanlı olarak metne dönüştüren web uygulaması.  
[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) ile transkripsiyon, [Resemblyzer](https://github.com/resemble-ai/resemblyzer) ile konuşmacı ayrıştırma ve [Gradio](https://www.gradio.app/) ile arayüz sağlar.

## Özellikler

- Canlı akış — segmentler tamamlandıkça ekrana yansır
- Model seçimi — small (hızlı), medium (dengeli), large-v3 (kaliteli)
- Konuşmacı ayrıştırma — her konuşmacı ayrı paragraf olarak etiketlenir (token gerektirmez)
- Otomatik konuşmacı sayısı tespiti (1–10 arası)
- CUDA varsa GPU, yoksa CPU ile çalışır
- Transkripsiyon tamamlandığında `.txt` olarak indirilir
- Duraklat / Devam et / Durdur desteği

## Kurulum

```bash
uv sync
```

> İlk çalıştırmada Whisper modeli (~500 MB) ve Resemblyzer encoder'ı (~17 MB) Hugging Face Hub'dan otomatik indirilir.

## Kullanım

```bash
uv run python main.py
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
│   └── ui.py             # Gradio arayüzü ve callback'ler
├── main.py               # Giriş noktası
├── pyproject.toml
└── README.md
```

## Bağımlılıklar

| Paket | Görev |
| --- | --- |
| `faster-whisper` | Konuşmadan metne (CTranslate2 tabanlı) |
| `resemblyzer` | Konuşmacı gömme vektörleri |
| `scikit-learn` | KMeans kümeleme (konuşmacı ayrıştırma) |
| `torch` (CPU) | Resemblyzer için PyTorch arka ucu |
| `gradio` | Web arayüzü |
