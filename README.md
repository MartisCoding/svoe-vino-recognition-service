# Vino Service

Backend-сервис для распознавания вин по фотографии. Проект запускается через Docker Compose, который генерируется скриптом `compose.py` на основе JSON-конфигурации.

Архитектура состоит из:

* FastAPI backend
* PostgreSQL
* MinIO
* RabbitMQ
* Qdrant
* CV inference worker
* OCR inference worker
* Frontend

## Архитектура

```text
                        ┌───────────────┐
                        │    Frontend   │
                        │      :3000    │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │    Backend    │
                        │      :8080    │
                        └───────┬───────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
        ┌──────────┐      ┌───────────┐      ┌──────────┐
        │ Postgres │      │   MinIO   │      │ RabbitMQ │
        │  :5432   │      │ :9000/:9001│     │ :5672    │
        └──────────┘      └───────────┘      └─────┬────┘
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              │                                           │
                              ▼                                           ▼
                     ┌─────────────────┐                         ┌─────────────────┐
                     │ CV Inference    │                         │ OCR Inference   │
                     │    worker       │                         │     worker      │
                     │      GPU        │                         │       GPU       │
                     └────────┬────────┘                         └────────┬────────┘
                              │                                           │
                              ▼                                           │
                         ┌──────────┐                                     │
                         │  Qdrant  │                                     │
                         │  :6333   │                                     │
                         └──────────┘                                     │
                              │                                           │
                              └───────────────────┬───────────────────────┘
                                                  ▼
                                               Backend
```

## Требования

Для запуска необходимы:

* Docker
* Docker Compose
* NVIDIA Container Toolkit
* NVIDIA GPU для `inference` и `ocr-inference`
* MinIO AIStor license

Проверить Docker:

```bash
docker --version
docker compose version
```

Проверить доступность GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu24.04 nvidia-smi
```

## Структура проекта

```text
.
├── backend/
│   ├── ...
│   └── .dockerfile
├── inference/
│   ├── ...
│   └── .dockerfile
├── ocr_inference/
│   ├── ...
│   └── .dockerfile
├── frontend/
│   ├── ...
│   └── Dockerfile
├── licenses/
│   └── minio.license
├── docker-config.json
├── docker-compose.generated.yml
├── compose.py
└── README.md
```

## Конфигурация

Основная конфигурация находится в:

```text
docker-config.json
```

`compose.py` содержит встроенные значения по умолчанию. Значения из `docker-config.json` накладываются поверх них.

Таким образом:

```text
DEFAULTS
   │
   ▼
docker-config.json
   │
   ▼
merged configuration
   │
   ▼
docker-compose.generated.yml
```

Можно получить полный конфиг по умолчанию:

```bash
python compose.py --generate-config
```

## Генерация Compose

Обычный запуск:

```bash
python compose.py
```

Скрипт:

1. Загружает `docker-config.json`, если он существует.
2. Объединяет его с `DEFAULTS`.
3. Генерирует `docker-compose.generated.yml`.
4. При необходимости останавливает все запущенные Docker-контейнеры.
5. Выполняет `docker compose up -d --build`.

По умолчанию используются:

```text
project_name: vino-service
detach: true
build: true
stop_all_before_build: true
```

## Полезные параметры

### Только сгенерировать Compose

```bash
python compose.py --dry-run
```

Скрипт сгенерирует:

```text
docker-compose.generated.yml
```

и покажет команду запуска, но не будет запускать Docker.

### Не собирать образы

```bash
python compose.py --no-build
```

### Запустить Compose в foreground

```bash
python compose.py --no-detach
```

### Не останавливать существующие контейнеры

```bash
python compose.py --skip-stop-all-before-build
```

### Использовать другой конфиг

```bash
python compose.py --config my-config.json
```

### Использовать другой путь для сгенерированного Compose

```bash
python compose.py --compose-output compose.yml
```

Параметры можно комбинировать:

```bash
python compose.py \
    --config docker-config.json \
    --compose-output docker-compose.generated.yml \
    --skip-stop-all-before-build
```

## Запуск вручную

Сгенерированный Compose-файл можно запускать напрямую:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    up -d
```

С пересборкой:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    up -d --build
```

Остановка:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    down
```

Проверка состояния:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    ps
```

Логи:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    logs -f
```

Логи конкретного сервиса:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    logs -f backend
```

## Сервисы

| Сервис     | Описание              |    Порт |
| ---------- | --------------------- | ------: |
| `postgres` | Основная БД           |  `5432` |
| `minio`    | Хранилище изображений |  `9000` |
| `minio`    | Web Console           |  `9001` |
| `rabbitmq` | Очередь сообщений     |  `5672` |
| `rabbitmq` | Management UI         | `15672` |
| `qdrant`   | Векторная БД          |  `6333` |
| `qdrant`   | gRPC                  |  `6334` |
| `backend`  | FastAPI               |  `8080` |
| `frontend` | Frontend              |  `3000` |

## Backend

Backend запускается на:

```text
http://localhost:8080
```

OpenAPI:

```text
http://localhost:8080/docs
```

Основные зависимости:

```text
PostgreSQL
MinIO
RabbitMQ
```

Backend использует:

```text
DATABASE
MINIO
RABBITMQ
PARSER
LOGGING
```

Каталог вин:

```text
./catalog.jsonl
```

## Frontend

Frontend доступен на:

```text
http://localhost:3000
```

Параметры Vite передаются в Docker build arguments:

```text
VITE_API_BASE_URL
VITE_DEMO_MODE
VITE_FESTIVAL_ENABLED
VITE_FESTIVAL_CHANCE
VITE_FESTIVAL_URL
VITE_FEEDBACK_ENABLED
```

Текущая конфигурация:

```text
VITE_API_BASE_URL=
VITE_DEMO_MODE=false
VITE_FESTIVAL_ENABLED=false
VITE_FESTIVAL_CHANCE=0.25
VITE_FEEDBACK_ENABLED=false
```

## PostgreSQL

Используется:

```text
PostgreSQL 16 Alpine
```

Параметры по умолчанию:

```text
Host: localhost
Port: 5432

Database: mydb
User: postgres
Password: postgres
```

Данные сохраняются в Docker volume:

```text
postgres_data
```

## MinIO

Используется MinIO AIStor:

```text
quay.io/minio/aistor/minio:latest
```

API:

```text
http://localhost:9000
```

Console:

```text
http://localhost:9001
```

Учётные данные:

```text
User: minioadmin
Password: minioadmin
```

Bucket для изображений:

```text
images
```

Данные хранятся в:

```text
minio_data
```

Для запуска AIStor требуется license:

```text
./licenses/minio.license
```

Этот каталог монтируется в контейнер как:

```text
/licenses
```

## RabbitMQ

RabbitMQ используется для обмена сообщениями между backend и inference workers.

Management UI:

```text
http://localhost:15672
```

Учётные данные по умолчанию:

```text
User: guest
Password: guest
```

Virtual host:

```text
/
```

Данные:

```text
rabbitmq_data
```

### Очереди CV

```text
wine.recognition.requests
wine.recognition.results
```

### Очереди OCR

```text
ocr.recognition.tasks
ocr.recognition.results
```

Схема работы:

```text
Backend
   │
   ├── wine.recognition.requests ──► CV worker
   │                                  │
   │                                  ▼
   │                         wine.recognition.results
   │                                  │
   │                                  ▼
   │                               Backend
   │
   └── ocr.recognition.tasks ─────► OCR worker
                                      │
                                      ▼
                              ocr.recognition.results
                                      │
                                      ▼
                                   Backend
```

## Qdrant

Используется для хранения и поиска wine embeddings.

REST API:

```text
http://localhost:6333
```

gRPC:

```text
localhost:6334
```

Collection:

```text
wines
```

Данные сохраняются в:

```text
qdrant_data
```

## CV Inference Worker

Сервис:

```text
inference
```

Использует NVIDIA GPU.

Основные параметры:

```text
Worker:
    cv_inference_worker

Device:
    cuda:0

Embedding size:
    512

Image size:
    224
```

Модели:

```text
/app/models/ArcFace_v2.pth
/app/models/YOLO_best_100.pt
```

Каталог датасета:

```text
/app/dataset/train
```

Qdrant:

```text
Host: qdrant
Port: 6333
Collection: wines
```

Prefetch:

```text
1
```

## OCR Inference Worker

Сервис:

```text
ocr-inference
```

Также использует NVIDIA GPU.

Основные параметры:

```text
Worker:
    ocr_inference_worker

Device:
    cuda:0

CRAFT canvas:
    2560

Recognition passes:
    3

Multilingual:
    enabled

Top K:
    5
```

Каталог:

```text
/app/dataset/wine_catalog.jsonl
```

PARSeq checkpoint:

```text
/app/weights/parseq_wine_best.pt
```

## Docker Volumes

Проект использует следующие volumes:

```text
postgres_data
minio_data
rabbitmq_data
qdrant_data
```

Посмотреть volumes:

```bash
docker volume ls
```

Посмотреть конкретный volume:

```bash
docker volume inspect vino-service_postgres_data
```

Удаление volumes приведёт к потере сохранённых данных:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    down -v
```

Использовать эту команду осторожно.

## Проверка сервисов

Проверить контейнеры:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    ps
```

Проверить backend:

```bash
curl http://localhost:8080/docs
```

Проверить Qdrant:

```bash
curl http://localhost:6333/readyz
```

Проверить MinIO Console:

```text
http://localhost:9001
```

Проверить RabbitMQ:

```text
http://localhost:15672
```

## Пересборка

Полная пересборка:

```bash
python compose.py
```

Или вручную:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    build --no-cache
```

После сборки:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    up -d
```

## Очистка

Остановить проект:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    down
```

Удалить контейнеры и volumes:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    down -v
```

Удалить неиспользуемые Docker-ресурсы:

```bash
docker system prune
```

Для удаления также неиспользуемых образов:

```bash
docker system prune -a
```

## Изменение конфигурации

Рекомендуется изменять:

```text
docker-config.json
```

а не редактировать:

```text
docker-compose.generated.yml
```

После изменения конфигурации необходимо заново выполнить:

```bash
python compose.py
```

Пример изменения порта backend:

```json
{
  "backend": {
    "port": 8081
  }
}
```

После генерации:

```text
localhost:8081
        │
        ▼
container:8080
```

## Безопасность

Текущая конфигурация содержит development credentials:

```text
postgres / postgres
minioadmin / minioadmin
guest / guest
```

Перед использованием в production необходимо изменить пароли и, по возможности, вынести секреты из обычного JSON-конфига.

Также не следует публиковать:

```text
docker-config.json
```

если в нём находятся реальные production credentials.

## Типичный workflow разработки

Изменить конфигурацию:

```text
docker-config.json
```

Сгенерировать Compose:

```bash
python compose.py --dry-run
```

Проверить:

```text
docker-compose.generated.yml
```

Запустить:

```bash
python compose.py
```

Проверить состояние:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    ps
```

Посмотреть логи:

```bash
docker compose \
    -f docker-compose.generated.yml \
    -p vino-service \
    logs -f
```

## Примечания

`docker-compose.generated.yml` является производным файлом и генерируется автоматически из:

```text
DEFAULTS
+
docker-config.json
```

Источником истины для инфраструктурных параметров является `docker-config.json`, а логика генерации находится в `compose.py`.
