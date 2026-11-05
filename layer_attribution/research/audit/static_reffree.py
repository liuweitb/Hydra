"""Static check: reference_free features must not touch `ref`, reference-based helpers, or module-level mutable state."""
import ast, sys, pathlib
src_path = pathlib.Path("D:/Hydra/layer_attribution/src/layer_attribution/features.py")
src = src_path.read_text(encoding="utf-8")
tree = ast.parse(src)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
def deco_info(fn):
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "feat":
            kind = d.args[0].value
            layer = d.args[1].value
            return kind, layer
    return None
reg = {n: deco_info(f) for n, f in funcs.items() if deco_info(f)}
rb_names = {n for n, (k, l) in reg.items() if k == "reference_based"}
rf_names = {n for n, (k, l) in reg.items() if k == "reference_free"}
def names_used(fn):
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
def attr_chain(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out
def closure(name, seen=None):
    seen = set() if seen is None else seen
    if name in seen or name not in funcs: return seen
    seen.add(name)
    for u in names_used(funcs[name]):
        if u in funcs: closure(u, seen)
    return seen
module_names = set()
for n in tree.body:
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name): module_names.add(t.id)
print("reference_free features:", len(rf_names), " reference_based:", len(rb_names))
bad = 0
for name in sorted(rf_names):
    cl = closure(name)
    ref_uses = []
    sig = [a.arg for a in funcs[name].args.args]
    for c in sorted(cl):
        fn = funcs[c]
        used = names_used(fn)
        argnames = [a.arg for a in fn.args.args]
        if c == name:
            # in the feature itself, `ref` is its own parameter (None); flag any use in the body
            body_names = set()
            for stmt in fn.body:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name): body_names.add(n.id)
            if "ref" in body_names: ref_uses.append((c, "uses ref in body"))
        else:
            if "ref" in argnames: ref_uses.append((c, "helper takes ref"))
            if "ref" in used and "ref" not in argnames: ref_uses.append((c, "helper uses free name ref"))
        # calls to reference-based features
        hit = used & rb_names
        if hit: ref_uses.append((c, f"calls reference_based {sorted(hit)}"))
    mods = sorted((names_used(funcs[name]) & module_names))
    print(f"{name:34s} sig={sig} closure={sorted(cl - {name})} module_consts={mods} REF_FLAGS={ref_uses}")
    if ref_uses: bad += 1
print("reference_free features with ref flags:", bad)
# any use of global state mutation
for n in ast.walk(tree):
    if isinstance(n, ast.Global): print("global statement at line", n.lineno)
