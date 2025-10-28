# 🚀 FastAPI + LangChain Demo: LLM-Agents API

Пример учебного backend-сервиса на **FastAPI** с интеграцией **LangChain**-агентов.
Проект демонстрирует, как создать минимальное API для взаимодействия с несколькими типами LLM-агентов — от простого чат-бота до ReAct-агента с инструментами и суммаризатора текстов.

---

## 🧭 Основная идея

Цель — показать студентам, как проектировать и запускать backend-приложения для LLM-агентов:

- работа с FastAPI: маршруты (`Routers`), `Middleware`, загрузка файлов;
- структура проекта по модулям (`core/`, `agents/`, `routers/`);
- подключение LLM через `LangChain` и конфигурацию `.env`;
- пример ReAct-агента, способного использовать инструменты (`Tools`) для выполнения действий.

---

## 📁 Структура проекта

```
fastapi_llm_demo/
├── main.py                 # Точка входа FastAPI
├── config.py               # Конфигурация через Pydantic Settings (.env)
├── core/
│   └── llm.py              # Инициализация LLM и базовая цепочка
├── agents/
│   ├── registry.py         # Реестр доступных агентов
│   ├── chat_basic.py       # Простой чат-агент (prompt → ответ)
│   ├── chat_react.py       # ReAct-агент с инструментами (LangChain 0.3.x)
│   ├── summarizer.py       # Агент для суммаризации текста
│   └── tools.py            # Инструменты (time, word_count и др.)
├── routers/
│   ├── chat.py             # REST API для взаимодействия с агентами
│   ├── files.py            # Загрузка и обработка файлов
│   └── health.py           # Проверка состояния API
├── requirements.txt        # Зависимости
├── .env.example            # Пример конфигурации
└── README.md               # (вы читаете его)
```

---

## ⚙️ Установка и запуск

### 1. Клонировать проект
```bash
git clone https://github.com/<yourname>/fastapi_llm_demo.git
cd fastapi_llm_demo
```

### 2. Создать виртуальное окружение
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Настроить `.env`
Скопировать шаблон и вписать свой API-ключ:
```bash
cp .env .env
source .env
```

Пример содержимого:
```env
OPENAI_API_KEY2=sk-...
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
TIMEOUT_S=60
```

### 5. Запустить приложение
```bash
uvicorn main:app --reload
```
или
```bash
python main.py
```

Откройте документацию Swagger UI:
👉 [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## 🧩 API-эндпоинты

| Метод | Эндпоинт | Описание |
|-------|-----------|-----------|
| `GET` | `/` | Проверка работы сервиса |
| `GET` | `/health/` | Технический статус API |
| `GET` | `/chat/agents` | Список доступных агентов |
| `POST` | `/chat/` | Отправка промпта агенту (суммаризация или рубрикация) |
| `POST` | `/files/upload-and-summarize` | Загрузка `.txt`/`.md` файла и краткое суммирование |
| `POST` | `/files/upload-and-categorize` | Загрузка `.txt`/`.md`/`.json`/`.pdf` и рубрикация |

---

## 🧠 Примеры использования

### Пример 1 — Суммаризация текста через `/chat`
```json
POST /chat
{
  "user_prompt": "Summarize the following notes about transformer architectures...",
  "agent": "summarizer"
}
```

**Ответ:**
```json
{
  "agent": "summarizer",
  "response": "- Transformers rely on self-attention..."
}
```

### Пример 2 — Рубрикация текста через `/chat`
```json
POST /chat
{
  "user_prompt": "Categorize this research abstract on quantum networking...",
  "agent": "categorizer"
}
```

**Ответ:**
```json
{
  "agent": "categorizer",
  "response": "{\"primary_domain\": \"Quantum Networking\", ...}"
}
```

### Пример 3 — Суммаризация файла
```bash
curl -X POST "http://127.0.0.1:8001/files/upload-and-summarize"      -F "file=@lecture_notes.txt"
```

**Ответ:**
```json
{
  "filename": "lecture_notes.txt",
  "summary": "- Key topic: LLM agent orchestration\n- Main idea: Function calling and ReAct\n..."
}
```

### Пример 4 — Рубрикация PDF-файла
```bash
curl -X POST "http://127.0.0.1:8001/files/upload-and-categorize" \
     -F "file=@paper.pdf"
```

**Ответ:**
```json
{
  "filename": "paper.pdf",
  "categorization": "{\"primary_domain\": \"Photonics\", ...}"
}
```

---

## 🧰 Агентная система

| Агент | Назначение | Особенности |
|-------|-------------|-------------|
| **summarizer** | Обработка текстов | Суммаризация загруженных документов |
| **categorizer** | Рубрикация научно-технических материалов | Возвращает структурированную классификацию |

---

## 🧩 Middleware и логирование

Каждый запрос оборачивается в middleware, который добавляет в заголовки время выполнения:
```
X-Process-Time: 0.183
```

Можно расширить логику: сохранять логи в файл, реализовать ограничение скорости или метрики OpenTelemetry.

---

## 🧪 Тестирование (опционально)

```bash
pytest -q
```

Пример теста можно добавить в `tests/test_chat.py`:
```python
def test_health(client):
    r = client.get("/health/")
    assert r.status_code == 200
    assert "ok" in r.json()["status"]
```

---

## 📘 Образовательные цели

Проект используется в рамках дисциплины
**«Методы разработки ПО с использованием LLM» (МИФИ, ИИКС)**

**Цель:**
Научить студентов строить минимальные backend-сервисы с интеграцией LLM-агентов и структурировать код по промышленным шаблонам (routers, core, config, agents).

---

## 📜 Лицензия

MIT License © 2025 — учебный демо-проект МИФИ / Институт интеллектуальных кибернетических систем
Автор: [Даниил Сухоруков](https://github.com/dsuhoi)

---

> 💡 Совет для студентов:
> Попробуйте добавить нового агента — например, переводчика (`translator.py`) или анализа CSV-файлов (`data_analyst.py`).
> Это отличный первый шаг к собственному проекту на FastAPI + LLM.
