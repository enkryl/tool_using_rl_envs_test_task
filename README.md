# ДЗ4: LLM-агент для анализа нарушений в метро

Многошаговая среда (ToolEnv) для LLM-агента, который анализирует данные о нарушениях в метро с помощью SQL-инструментов. Обучение через GRPO (Group Relative Policy Optimization).

## Быстрый старт

### 1. Генерация данных (локально)

```bash
cd d:\omut\hw4

# Smoke-тест (5 эпизодов на каждый уровень сложности)
python scripts/generate_data.py --smoke

# Полная генерация (~2350 train + 500 eval)
python scripts/generate_data.py
```

### 2. Бейзлайн (рекомендуется GPU)

```bash
# Один eval-бакет
python agent/run_agent.py --model baseline --data data/eval_d1.jsonl --verbose

# Все бакеты
for i in 1 2 3 4 5; do
    python agent/run_agent.py --model baseline --data data/eval_d${i}.jsonl --output-dir logs
done
```

### 3. Обучение GRPO (требуется GPU)

```bash
pip install -r requirements_gpu.txt

# Через скрипт
python training/grpo_train.py --epochs 3 --batch-size 2 --num-generations 4

# Или через ноутбук (рекомендуется)
jupyter notebook training/train_notebook.ipynb
```

### 4. Оценка и сравнение

```bash
# Прогон обученной модели на eval
for i in 1 2 3 4 5; do
    python agent/run_agent.py --model grpo --model-path checkpoints/grpo/final --data data/eval_d${i}.jsonl
done

# Сравнение результатов
python scripts/compare_results.py --baseline-dir logs --grpo-dir logs
```

## Структура проекта

```
base/           # Data, ToolEnv, TrajectoryVerifier (абстрактные классы)
env/            # Генератор БД, инструменты, MetroViolationsEnv, генератор эпизодов
verifier/       # MetroTrajectoryVerifier
agent/          # BaselineAgent + скрипт запуска
training/       # GRPO-тренер, функция награды, ноутбук
scripts/        # Генерация данных, оценка, сравнение
data/           # Сгенерированные датасеты
logs/           # Логи обучения, метрики, траектории
```

## Детали среды

- **10 уровней сложности**: d1 (простой подсчёт) → d10 (полный workflow с открытием кейса)
- **7 инструментов**: get_schema, get_table_sample, run_sql, lookup_station, lookup_violation, get_case, open_case
- **Policy rules**: подтверждение перед мутациями, запрет на галлюцинацию сущностей
- **Reward**: outcome (+1/−1) + shaping (штрафы за шаги/тулы/нарушения политик)
- **Curriculum learning**: эпоха 1 — d1–d4, эпоха 2 — d1–d7, эпоха 3 — d1–d10
