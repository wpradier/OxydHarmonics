# Oxyd Harmonics

## Launch app (MLP only works on linux):

### First build the library
```bash
cd modules/ml/ml_core
cargo build --release
```

### run the app

```bash
cd modules/ml/ml/models/mlp
jupyter notebook oxyd_prod.ipynb
```

App should be running on http://127.0.0.1:7860.