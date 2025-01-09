# app.py
import gradio as gr
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio
import numpy as np

# Инициализация модели
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")


def process_audio(audio_path, duration=10):
    """Обработка входного аудио"""
    if audio_path is None:
        return None

    try:
        # Загрузка и ресемплирование аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 32000:
            resampler = torchaudio.transforms.Resample(sample_rate, 32000)
            waveform = resampler(waveform)

        # Конвертация в моно если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.numpy()
    except Exception as e:
        raise gr.Error(f"Ошибка обработки аудио: {str(e)}")


def generate_music(audio_file, prompt, duration=10):
    """Генерация музыки на основе входного аудио и промпта"""
    try:
        # Обработка входного аудио
        if audio_file is not None:
            audio_array = process_audio(audio_file)
            inputs = processor(
                audio=audio_array,
                sampling_rate=32000,
                text=prompt,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )

        # Генерация
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 50,  # Примерное количество токенов для заданной длительности
                do_sample=True,
                guidance_scale=3.0
            )

        # Конвертация в numpy и нормализация
        audio_array = audio_values.cpu().numpy().squeeze()

        return (32000, audio_array)

    except Exception as e:
        raise gr.Error(f"Ошибка генерации: {str(e)}")


# Создание интерфейса
with gr.Blocks() as demo:
    gr.Markdown("# Генератор музыки на основе референса")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Загрузите референсный трек (опционально)",
                type="filepath"
            )
            prompt = gr.Textbox(
                label="Текстовый промпт",
                placeholder="Опишите желаемую музыку"
            )
            duration = gr.Slider(
                minimum=5,
                maximum=30,
                value=10,
                step=5,
                label="Длительность (секунды)"
            )
            generate_btn = gr.Button("Сгенерировать")

        with gr.Column():
            output_audio = gr.Audio(
                label="Сгенерированная музыка",
                type="numpy"
            )

    # Обработчик событий
    generate_btn.click(
        fn=generate_music,
        inputs=[audio_input, prompt, duration],
        outputs=output_audio
    )

# Запуск приложения
demo.launch()