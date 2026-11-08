# Проверка deterministic-val-loss

Дата: 2026-11-09. Реализация, CPU-проверки и GPU-приёмка завершены;
см. [итоговый отчёт](validation_gpu_report.md).

## Локальные результаты

Все 48 тестов прошли в Windows, Python 3.12.11, PyTorch 2.7.0 CPU.
Использованы реальные классы тренера и обёрток моделей с малыми подставными
входами; полные веса моделей не загружались.

```powershell
.venv/Scripts/python.exe -m unittest testing.test_validation_config testing.test_validation_manifest testing.test_validation_inputs testing.test_validation_loss testing.test_validation_lifecycle testing.test_validation_conditions testing.test_validation_distributed testing.test_validation_isolation testing.test_validation_qwen_edit testing.test_validation_qwen21_loading -v
```

Проверены:

- Разбор конфигурации, поддержка входов по семейству модели и ранние ошибки.
- Манифест, альтернативы целей, фиксированные входы и автоматические контроли.
- Общий train/val loss, маски, дробные веса и чтение состояния обучаемого SNR.
- Восстановление RNG, градиентов, режимов модулей и следующего обновления.
- Накопление градиентов, пропуски обновлений, расписание val и TensorBoard.
- Два CPU-процесса: единственная публикация и передача ошибки без зависания.
- FP32 LoRA, парное состояние обучения, прерванная запись, resume, EMA и ротация.
- Загрузка полных HF-весов Qwen-2.1/Turbo из одного или нескольких файлов,
  строгая проверка параметров и совместимость BF16-входов с FP32 VAE.
- Qwen-Image-Edit-2511: обязательные контроли, один/несколько входов и их порядок
  в тексте и latent, маска, автоматический depth, альтернативы, оба значения
  `match_target_res`, общий train/val loss, повторный проход без VAE-кодирования,
  подготовка при resume и неизменность следующего обновления. Отдельно проверены
  восстановление RNG при ошибке и прежняя обработка обучающего батча из двух
  примеров. Генератор GPU-конфигов включает Edit-2511 с контролями и sample reference.

`openspec validate deterministic-val-loss --strict --json` прошёл без замечаний.
`git diff --check` прошёл. Основной `config/qwen_image_21_kibm.yaml` не изменён.

## GPU и готовность CLI

Пройдены 22 запуска initial/resume: 11 сценариев для 7 моделей с полными
неквантованными весами. Проверены CUDA RNG, повторный val, checkpoint и samples.
Численная проверка следующего обновления использует SDPA math; ограничение
быстрого backward описано отдельно в GPU-отчёте. Тестовые данные и лишние модели
удалены. Основной Qwen-2.1 загружен без сети и внешнего кэша; CLI готов,
основное обучение на 2500 шагов не запускалось.

Контрольные входы используются только у моделей, которые их поддерживают:
Qwen-Image-2.1, Qwen-Image-Edit-2511 и MOSS в edit-режиме. Для Qwen-Image/2512 и MOSS без edit
такая конфигурация отклоняется до обучения. Маска loss проверяется отдельно.
