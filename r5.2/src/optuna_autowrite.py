import ast
import atexit
import builtins
import os
import re
import shutil

_PRINTED = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(\S.*)$")


def autowrite_printed_config(target_path, rename=None):
    rename = rename or {}
    captured = {}
    real_print = builtins.print

    def spy(*args, **kwargs):
        real_print(*args, **kwargs)
        if kwargs.get("file") is None:
            for line in " ".join(str(a) for a in args).splitlines():
                m = _PRINTED.match(line)
                if m:
                    key = rename.get(m.group(1), m.group(1))
                    captured[key] = m.group(2).strip()

    builtins.print = spy

    targets = [target_path] if isinstance(target_path, str) else list(target_path)

    def flush():
        builtins.print = real_print
        if not captured:
            return
        files = {}
        for t in targets:
            if os.path.exists(t):
                with open(t) as f:
                    files[t] = f.readlines()
            else:
                real_print(f"[autowrite] target not found: {t}")
        written, skipped = {}, []
        for key, raw in captured.items():
            try:
                value = ast.literal_eval(raw)
            except Exception:
                skipped.append(f"{key} (unparsable value {raw!r})")
                continue
            pat = re.compile(rf"^{re.escape(key)}\s*=")
            hit = False
            for t, lines in files.items():
                for i, line in enumerate(lines):
                    if pat.match(line):
                        comment = "  #" + line.split("#", 1)[1].rstrip("\n") if "#" in line else ""
                        lines[i] = f"{key} = {value!r}{comment}\n"
                        written.setdefault(os.path.basename(t), []).append(key)
                        hit = True
                        break
            if not hit:
                skipped.append(f"{key} (not defined in any target)")
        for t, lines in files.items():
            if os.path.basename(t) in written:
                shutil.copyfile(t, t + ".bak_before_optuna")
                with open(t, "w") as f:
                    f.writelines(lines)
        for name, keys in written.items():
            real_print(f"\n[autowrite] {name} updated: {', '.join(keys)}")
        if skipped:
            real_print(f"[autowrite] NOT written: {'; '.join(skipped)}")

    atexit.register(flush)
