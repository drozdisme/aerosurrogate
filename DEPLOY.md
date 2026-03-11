# AeroSurrogate — Деплой с реальными ML моделями

## Что происходит при первом старте

```
docker-compose up
  ├── mlflow стартует (трекинг метрик)
  └── api стартует
        ├── Генерация датасета: 60k сэмплов (~2 мин)
        ├── Обучение XGBoost: Cl, Cd, Cm (~5 мин) → R² > 0.99
        ├── Обучение Cp surrogate (~3 мин)
        └── API готов → UI поднимается
```

**Первый старт: ~15-20 минут.** Последующие: < 10 секунд (модели в volume).

---

## Railway.app (рекомендуется, бесплатно)

### 1. GitHub репозиторий

```bash
cd aerosurrogate-v2
echo "dataset/\nmodels/*.pkl\nmodels/*.pt\n__pycache__/\n*.pyc" > .gitignore
git init && git add . && git commit -m "production deploy"
git remote add origin https://github.com/YOUR/aerosurrogate.git
git push -u origin main
```

### 2. Деплой

- [railway.app](https://railway.app) → Sign in with GitHub
- **New Project** → **Deploy from GitHub repo** → выберите репо
- Railway найдёт `docker-compose.yml` автоматически

### 3. Порты в Railway Dashboard

| Сервис | Порт | Доступ |
|--------|------|--------|
| `ui`     | 3000 | **Generate Domain** (публичный URL) |
| `api`    | 8000 | Internal (не открывать наружу) |
| `mlflow` | 5000 | Generate Domain (опционально) |

### 4. Первый деплой

Ждите ~20 минут. В логах `api` увидите:
```
  ✓ Training complete
  Metrics:
    Cl_R2: 0.9943
    Cd_R2: 0.9921
    Cm_R2: 0.9887
  Starting API on :8000
```
После этого UI доступен по Railway URL.

---

## Render.com (альтернатива)

На Render сервисы изолированы — внутренний Docker DNS не работает.
Нужно деплоить `api` и `ui` как отдельные Web Services.

**API:**
- New → Web Service → Dockerfile: `Dockerfile.api` → Port: `8000`
- Health Check Grace Period: **1200 секунд** (иначе убьёт до окончания обучения)

**UI — обновите nginx.conf** перед деплоем:
```nginx
# Замените:
proxy_pass  http://api:8000/;
# На ваш Render URL:
proxy_pass  https://aerosurrogate-api.onrender.com/;
```

- New → Web Service → Dockerfile: `Dockerfile.ui` → Port: `3000`

⚠️ Бесплатный Render засыпает после 15 мин — платный план $7/мес для демо.

---

## Параметры обучения

Устанавливаются как переменные окружения сервиса `api`:

| Переменная | По умолчанию | Описание |
|-----------|-------------|---------|
| `N_SAMPLES` | `60000` | `30000` быстрее, `100000` точнее |
| `SKIP_FNO` | `1` | `0` = включить FNO (+30 мин) |

## Переобучение (если изменили параметры)

```bash
docker-compose down -v && docker-compose up --build
```

## Проверка

```bash
curl https://your-api-url/health
# {"status":"ok","demo_mode":false,"metrics":{"Cl_R2":0.994,...}}
```

Swagger: `https://your-api-url/docs`
