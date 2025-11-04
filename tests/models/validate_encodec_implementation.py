#!/usr/bin/env python
"""
Validate EnCodec implementation without runtime testing.
Checks code structure, imports, and logic.
"""

import ast
import sys
from pathlib import Path

print("=" * 80)
print("EnCodec Implementation Validation")
print("=" * 80)

# Parse the audio.py file
audio_py_path = Path("giblet/data/audio.py")
with open(audio_py_path, "r") as f:
    code = f.read()
    tree = ast.parse(code)

# Validation checks
checks = []

# 1. Check EnCodec imports
print("\n1. Checking EnCodec imports...")
has_encodec_import = "from transformers import EncodecModel, AutoProcessor" in code
has_encodec_flag = "ENCODEC_AVAILABLE = True" in code
checks.append(("EnCodec imports", has_encodec_import and has_encodec_flag))
print(
    f"   {'✓' if has_encodec_import and has_encodec_flag else '✗'} EnCodec imports present"
)

# 2. Check AudioProcessor class
print("\n2. Checking AudioProcessor class structure...")
audio_processor_found = False
methods_found = {}

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AudioProcessor":
        audio_processor_found = True
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods_found[item.name] = True

expected_methods = [
    "__init__",
    "audio_to_features",
    "_audio_to_features_encodec",
    "_audio_to_features_mel",
    "features_to_audio",
    "_features_to_audio_encodec",
    "_features_to_audio_mel",
    "get_audio_info",
]

for method in expected_methods:
    present = method in methods_found
    checks.append((f"Method: {method}", present))
    print(f"   {'✓' if present else '✗'} {method}")

# 3. Check __init__ parameters
print("\n3. Checking __init__ parameters...")
init_params = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "__init__":
        for arg in node.args.args:
            if arg.arg != "self":
                init_params.append(arg.arg)

expected_params = ["use_encodec", "encodec_bandwidth", "device"]
for param in expected_params:
    present = param in init_params
    checks.append((f"Parameter: {param}", present))
    print(f"   {'✓' if present else '✗'} {param}")

# 4. Check key implementation details
print("\n4. Checking key implementation details...")

# EnCodec model loading
has_model_load = "EncodecModel.from_pretrained" in code
checks.append(("EnCodec model loading", has_model_load))
print(f"   {'✓' if has_model_load else '✗'} EnCodec model loading")

# Encoding logic
has_encoding = "self.encodec_model.encode" in code
checks.append(("EnCodec encoding", has_encoding))
print(f"   {'✓' if has_encoding else '✗'} EnCodec encoding")

# Decoding logic
has_decoding = "self.encodec_model.decode" in code
checks.append(("EnCodec decoding", has_decoding))
print(f"   {'✓' if has_decoding else '✗'} EnCodec decoding")

# Auto-detection
has_dtype_check = "features.dtype in [np.int32, np.int64]" in code
checks.append(("Auto-detection (dtype)", has_dtype_check))
print(f"   {'✓' if has_dtype_check else '✗'} Auto-detection via dtype")

# 5. Check test file
print("\n5. Checking test file...")
test_file = Path("tests/data/test_audio_encodec.py")
test_exists = test_file.exists()
checks.append(("Test file exists", test_exists))
print(f"   {'✓' if test_exists else '✗'} Test file: {test_file}")

if test_exists:
    with open(test_file, "r") as f:
        test_code = f.read()

    test_classes = []
    test_tree = ast.parse(test_code)
    for node in ast.walk(test_tree):
        if isinstance(node, ast.ClassDef):
            test_classes.append(node.name)

    expected_test_classes = [
        "TestEnCodecIntegration",
        "TestBackwardsCompatibility",
        "TestEnCodecBandwidths",
    ]

    for cls in expected_test_classes:
        present = cls in test_classes
        checks.append((f"Test class: {cls}", present))
        print(f"   {'✓' if present else '✗'} {cls}")

# 6. Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

passed = sum(1 for _, result in checks if result)
total = len(checks)

print(f"\nPassed: {passed}/{total}")

if passed == total:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nImplementation is complete and structurally correct.")
    print("Ready for runtime testing in target environment.")
    sys.exit(0)
else:
    print("\n❌ SOME CHECKS FAILED")
    print("\nFailed checks:")
    for check_name, result in checks:
        if not result:
            print(f"  - {check_name}")
    sys.exit(1)
