"""
Command-line prediction utility.

Usage:
    python predict.py path/to/xray.png
"""
import sys

import model_utils
import remedies


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        result = model_utils.predict_image(path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n=== Prediction ===")
    print(f"  Label            : {result['label']}")
    print(f"  Confidence       : {result['confidence']}%")
    print(f"  P(Normal)        : {result['normal_probability']}%")
    print(f"  P(Tuberculosis)  : {result['tb_probability']}%")

    g = remedies.get_guidance(result["label"])
    print(f"\n{g['summary']}\n")
    print("Immediate steps:")
    for s in g["immediate_steps"]:
        print(f"  - {s}")
    print(f"\n{g['disclaimer']}")


if __name__ == "__main__":
    main()
