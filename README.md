# Rusty - Car Rust Classifier

Binary rust detector (Fine-Tuned PyTorch ResNet‑18). Given a car photo, returns `("rust" | "clean", probability)`.

## Install
```bash
pip install torch torchvision pillow
```

## Usage
```python
from rusty import predict_image

label, p = predict_image(
    "examples/some_car.jpg",
    model_path="models/rusty55_small.pth",
    threshold=0.55,
)
print(label, f"{p:.3f}")
```

## Notes
- Place your `.pth` at `models/rusty55_small.pth` or pass `model_path`.
- `threshold` ∈ [0,1]; higher = stricter about calling rust.
- Uses CPU or CUDA automatically.
