"""P0-1 冒烟测试：官方 open_clip 能否在本机 CPU 下导入并前向。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

t0 = time.time()
print("[1] importing open_clip ...", flush=True)
import open_clip  # noqa: E402
print(f"    open_clip {open_clip.__version__} imported OK ({time.time()-t0:.1f}s)", flush=True)

print("[2] creating model from local weights (CPU) ...", flush=True)
wd = Path("data/external/models/biomedclip")
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    pretrained=str(wd / "open_clip_pytorch_model.bin"),
)
model = model.to("cpu").eval()
print("    model created OK", flush=True)

print("[3] one forward pass (CPU) ...", flush=True)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
img = preprocess(Image.new("RGB", (224, 224), (128, 128, 128))).unsqueeze(0).to("cpu")
tok = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
ids = tok(["a chest x ray of the lungs"]).to("cpu")
with torch.no_grad():
    ie = model.encode_image(img)
    te = model.encode_text(ids)
print("    encode_image", tuple(ie.shape), "encode_text", tuple(te.shape), flush=True)
print("[OK] open_clip CPU forward works", flush=True)
