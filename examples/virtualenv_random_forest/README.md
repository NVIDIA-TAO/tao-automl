# Virtualenv Random Forest Smoke Test

This example runs two TAO AutoML recommendations against scikit-learn's
`RandomForestClassifier` and bundled Iris dataset. The training action executes
directly with the selected virtual environment's Python interpreter.

```bash
python -m venv /tmp/tao-automl-model-venv
/tmp/tao-automl-model-venv/bin/python -m pip install scikit-learn
python examples/virtualenv_random_forest/run.py \
  --venv-path /tmp/tao-automl-model-venv \
  --work-dir /tmp/tao-automl-virtualenv-smoke
```

The orchestrator Python environment must have the local `nvidia-tao-automl`
and `nvidia-tao-sdk` packages installed. The selected model environment only
needs scikit-learn.
