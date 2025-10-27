import sys, pkg_resources
print("✅ Python:", sys.version)
print("✅ Environment:", sys.prefix)

required = [
    "fastapi", "uvicorn", "pandas", "numpy", "scikit-learn",
    "aiohttp", "redis", "prometheus-client", "optuna",
    "lightgbm", "xgboost", "ccxt"
]

for pkg in required:
    try:
        v = pkg_resources.get_distribution(pkg).version
        print(f"✔ {pkg:<20} {v}")
    except Exception as e:
        print(f"❌ {pkg:<20} NOT FOUND ({e})")
