import importlib
import os
import sys


os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))


REQUIRED_PACKAGES = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "transformers": "4.44.2",
    "timm": "1.0.9",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "sklearn": "1.9.0",
    "scipy": "1.17.1",
    "matplotlib": "3.11.1",
    "PIL": "12.3.0",
    "yaml": "6.0.3",
    "tqdm": "4.70.0",
    "einops": "0.8.2",
    "nltk": "3.10.3",
    "openpyxl": "3.1.5",
    "psutil": "7.2.2",
    "regex": "2026.7.19",
    "sentencepiece": "0.2.2",
    "wandb": "0.28.2",
    "pytest": "9.1.1",
}

DISPLAY_NAMES = {
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


def check_python():
    version = sys.version_info

    if (version.major, version.minor) == (3, 11):
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True

    print(
        f"[FAIL] Python 3.11 required, "
        f"found {version.major}.{version.minor}.{version.micro}"
    )
    return False


def check_package(package_name, expected_version):
    display_name = DISPLAY_NAMES.get(package_name, package_name)

    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, "__version__", None)

        # PyTorch CPU/CUDA builds report versions such as 2.4.1+cpu.
        if package_name in ("torch", "torchvision"):
            version_matches = installed_version.startswith(expected_version)
        else:
            version_matches = installed_version == expected_version

        if version_matches:
            print(
                f"[OK] {display_name:<15} "
                f"{installed_version}"
            )
            return True

        print(
            f"[FAIL] {display_name:<15} "
            f"expected {expected_version}, "
            f"found {installed_version}"
        )
        return False

    except Exception as error:
        print(
            f"[FAIL] {display_name:<15} "
            f"import failed: {error}"
        )
        return False


def check_pytorch():
    try:
        import torch
    except Exception as error:
        print()
        print("PyTorch runtime:")
        print(f"  unavailable: {error}")
        return False

    print()
    print("PyTorch runtime:")
    print(f"  Version          : {torch.__version__}")
    print(f"  CUDA available   : {torch.cuda.is_available()}")
    print(f"  CUDA version     : {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU              : CPU")

    return True


def check_project_imports():
    project_modules = [
        "arguments",
        "models.cxr_encoder",
        "models.cxr_models",
        "models.ehr_encoder",
        "models.rr_encoder",
        "models.dn_encoder",
        "models.fusion",
        "datasets.cxr_dataset",
        "datasets.ehr_dataset",
        "datasets.DataFusion",
        "trainers.MSMA_trainer",
    ]

    original_path = list(sys.path)
    sys.path.insert(0, "medpatch")

    try:
        results = []
        for module_name in project_modules:
            try:
                importlib.import_module(module_name)
                print(f"[OK] {module_name}")
                results.append(True)
            except Exception as error:
                print(f"[FAIL] {module_name}: {error}")
                results.append(False)
        return all(results)
    finally:
        sys.path = original_path


def main():
    print("=" * 60)
    print("PneumoVision Environment Verification")
    print("=" * 60)

    print()
    print("Python:")
    python_ok = check_python()

    print()
    print("Required packages:")

    package_results = []

    for package, version in REQUIRED_PACKAGES.items():
        package_results.append(
            check_package(package, version)
        )

    pytorch_ok = check_pytorch()

    print()
    print("Project imports:")
    project_ok = check_project_imports()

    print()
    print("=" * 60)

    if python_ok and all(package_results) and pytorch_ok and project_ok:
        print("ENVIRONMENT READY")
        return 0

    print("ENVIRONMENT NOT READY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
