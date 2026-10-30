# Project documentation

Documentation for the diabetes risk detection capstone: a binary classifier that flags people for follow-up screening, from seven routine health indicators.

> **Not for clinical use.** The data is synthetic, and its label runs against clinical knowledge for 6 of 7 features. The service is a technical demonstration until the label definition is confirmed. See [Limitations](model_explanation.md#limitations).

## Where to start

| You want to... | Read | Written for |
|---|---|---|
| Understand what was built and what was found | [Project report](project_report.md) | Assessors and mentors |
| Present the project | [Presentation outline](presentation.md) | Capstone audience |
| Understand what the model does and when to trust it | [Model explanation](model_explanation.md) | Reviewers and non-specialists |
| Call the model from an application | [API reference](api_reference.md) | Developers integrating with it |
| Run the service | [Deployment guide](deployment_guide.md) | Whoever operates it |
| Change or extend the code | [Technical documentation](technical_documentation.md) | Engineers maintaining it |

The research behind the project, including the literature review and the full tables of results, is in the main [README](../README.md). The step-by-step data analysis is in [eda.ipynb](../eda.ipynb).

## The project in brief

| | |
|---|---|
| **Task** | Flag people at risk of diabetes for follow-up testing, prioritising recall |
| **Data** | 1,000 synthetic records; 997 after dropping under-18s; 15% positive |
| **Final model** | Logistic Regression, in two variants: with and without fasting glucose |
| **Test results** | Recall 0.87 in both. ROC-AUC 0.944 with glucose, 0.849 without |
| **Main finding** | Random Forest and Gradient Boosting did not beat the simple baseline |
| **Serving** | FastAPI service with `/health` and `/predict`, plus a Dockerfile |

## Quick start

```bash
pip install -r requirements.txt
python main.py                     # trains and writes models/ and reports/ (about 4 minutes)
uvicorn src.api:app --port 8000    # serves the models; interactive docs at /docs
```
