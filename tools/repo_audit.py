import ast
import json
import pathlib

ROOTS = [pathlib.Path("computations")]
summary = {}
for root in ROOTS:
    for p in root.glob("*.py"):
        code = p.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(p))
        funcs, classes = [], []
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                funcs.append(n.name)
            if isinstance(n, ast.ClassDef):
                classes.append(n.name)
        summary[p.stem] = {
            "path": str(p),
            "functions": funcs,
            "classes": classes,
            "lines": code.count("\n") + 1,
        }
pathlib.Path("reports/module_inventory.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("Zapisano reports/module_inventory.json")
