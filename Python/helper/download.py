from pathlib import Path

import torch
import torchvision.models.quantization as models


HELPER_DIR = Path(__file__).resolve().parent
OUT_PATH = HELPER_DIR / "mobilenet_V2.pth"


def main() -> None:
    quantized_mobilenet = models.mobilenet_v2(pretrained=True, quantize=True)
    torch.save(quantized_mobilenet.state_dict(), OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
