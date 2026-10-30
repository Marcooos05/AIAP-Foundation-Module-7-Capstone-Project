# Deployment guide

How to train, run and maintain the prediction service. For the endpoints, see the [API reference](api_reference.md).

> **Scope.** "Production-ready" here means the service is packaged and repeatable. It doesn't mean clinically valid. The model **must not inform real clinical decisions** until its label is confirmed ([Limitations](model_explanation.md#limitations)), and the service needs the additions under [Before real use](#before-real-use).

**Testing status.** Everything here was run on a development machine except the Docker steps. The `Dockerfile` has **not been built yet**, because no Docker daemon was available. Expect to check it on first use.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | Matches `.python-version` and the Docker image |
| `data/synthetic_data_166.csv` | Git-ignored, so it must be supplied |
| Several CPU cores to train | Training takes about 4 minutes on 16 cores, and longer on fewer |
| Docker | Optional, for the container |

Serving needs little: the models are about 8 KB each.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Versions are pinned. Keep scikit-learn the same between training and serving, because the saved pipelines may not load under another version.

## 2. Train

```bash
python main.py
```

The service needs two outputs, which must always stay together:

| File | Contains |
|---|---|
| `models/best_pipeline__{with,without}_glucose.joblib` | The fitted pipelines |
| `reports/selected_models.json` | The decision thresholds |

## 3. Run locally

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Check it:

```bash
curl http://localhost:8000/health
# {"status":"ok","models":{"with_glucose":true,"without_glucose":true}}
```

Open <http://localhost:8000/docs> to try requests in the browser.

## 4. Run in Docker

Train first: the image copies the models from your working folder, because `models/` is git-ignored.

```bash
python main.py
docker build -t diabetes-risk-api:1.0.0 .
docker run --rm -p 8000:8000 --name diabetes-risk-api diabetes-risk-api:1.0.0
```

The image is based on `python:3.11-slim`, runs as a non-root user, and includes a `HEALTHCHECK` that calls `/health`. Only `requirements.txt`, `src/`, `models/` and `reports/selected_models.json` go into the build, so the data, reports and tests stay out.

Check its health with:

```bash
docker inspect --format '{{.State.Health.Status}}' diabetes-risk-api
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Host and port | `uvicorn` flags | `0.0.0.0:8000` in Docker |
| Worker processes | `uvicorn --workers N` | 1 |
| Batch limit | `MAX_BATCH_SIZE` in `src/api.py` | 1,000 |
| Minimum age, levels, plausible ranges | `src/config.yaml` | See the [config reference](technical_documentation.md#configuration-reference) |
| Thresholds | `reports/selected_models.json`, set by training | 0.418 / 0.404 |

There are no environment variables or secrets. The API is stateless, so it can run behind a load balancer with several copies.

## Health checks and failure behaviour

| Situation | `/health` | `/predict` |
|---|---|---|
| Both models loaded | 200 `ok` | Works |
| One model missing | 503 `degraded` | Works for the other variant; 503 for the missing one |
| No models | 503 `degraded` | 503 |

Models load on the first request and are then kept in memory. If they're missing at startup, the service still starts and reports `degraded`. It picks them up without a restart once they appear.

Use `/health` for both readiness and liveness. With a load balancer, a 503 takes that copy out of rotation.

## Logging and monitoring

uvicorn logs each request's method, path and status to stdout. Request bodies aren't logged, and error responses don't repeat submitted values. There's no monitoring built in. Before real use, send logs to your central platform and track:

| Signal | Why | Example alert |
|---|---|---|
| 5xx rate and latency | Service health | Any sustained 503 |
| 422 rate | A client may be sending bad data | Sudden rise |
| Plausibility warnings per record | Input data may be drifting | Rate above the normal level |
| Share of records flagged | Model behaviour may be shifting | Far from the test rate of 28.5% with glucose, or 40% without |
| Input feature distributions | Drift from training data | Means moving by more than about 0.5 SD |

The flag rates are the test-set share of flagged records: 57 of 200 with glucose and 80 of 200 without.

## Retraining and updating

1. Back up the current `models/` and `reports/selected_models.json` together.
2. Update the data or config, then run `python main.py` and `python -m pytest`.
3. Review `reports/final_metrics.json`, `reports/selected_models.json` and `reports/feature_comparison.json`.
4. Restart the service, or build a new image with a new tag. A running service keeps its loaded models until restarted.
5. Check `/health` and send a known test record.

Don't re-use the test set to compare repeated retrains: results will gradually become optimistic. New data should provide a new test set.

## Versioning and rollback

- Treat each pipeline and threshold pair as one release. Never mix them across training runs.
- Tag images by version, such as `diabetes-risk-api:1.0.0`, and don't deploy `latest`.
- To roll back in Docker, redeploy the previous tag.
- To roll back without Docker, restore the backed-up files and restart.

## Security

The service has **no authentication, no rate limiting and no TLS**. Don't expose it to the internet as it stands.

- **Health data:** real inputs would be sensitive. The service stores nothing, but anything in front of it, such as a proxy logging request bodies, might.
- **Model files:** `.joblib` files are pickles and can run code when loaded. Only load ones from your own training runs.
- **Input limits:** batches are capped at 1,000 records, and unknown fields are rejected.
- **Dependencies:** versions are pinned, so scan and update them regularly.

## Before real use

**Clinical**

- [ ] Label definition confirmed with the data provider
- [ ] Retrained and validated on real data
- [ ] Performance assessed across patient groups
- [ ] Regulatory and ethics review done

**Engineering**

- [ ] Docker image built and tested
- [ ] Authentication and TLS, for example at an API gateway
- [ ] Rate limiting
- [ ] Central logging, monitoring and alerts
- [ ] Model artifacts stored in a registry, with versions
- [ ] CI that runs the tests on every change
- [ ] Load test at the expected request rate
