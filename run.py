#!/usr/bin/env python3
"""run.py -- the project's workflows: install, build, test, run.

  python3 run.py install            set up the reference stack for app_test.py
  python3 run.py build              compile app_main and app_test into build/
  python3 run.py test               run the C unit tests and the reference suite
  python3 run.py run -- <args>      build, then hand <args> to app_main
  python3 run.py bench --model DIR  build, then time prefill and decode
  python3 run.py clean              remove build/
  python3 run.py all                install, build, test

Build flavours
  --portable      target a widely available instruction set, not this host
  --no-simd       compile the scalar reference kernels only
  --debug         no optimisation, assertions on
  --sanitize      address and undefined behaviour sanitizers (clang or gcc)
  --cc PATH       use a specific compiler

Everything else after `--` goes straight to app_main.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
VENV = os.path.join(ROOT, ".venv")
NEEDS = ["torch", "transformers", "safetensors", "numpy", "tokenizers"]


# ---------------------------------------------------------------------------
# shell helpers
# ---------------------------------------------------------------------------

def say(line):
    print(f"\033[1m>\033[0m {line}", flush=True)


def call(cmd, **kw):
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd, **kw)


def venv_python():
    leaf = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    path = os.path.join(VENV, leaf)
    return path if os.path.exists(path) else sys.executable


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------

def find_cc(chosen):
    if chosen:
        return chosen
    if os.environ.get("CC"):
        return os.environ["CC"]
    for name in ("cc", "gcc", "clang", "cl"):
        found = shutil.which(name)
        if found:
            return found
    return None


def accepts(cc, flags):
    """True when the compiler builds a trivial file with these flags."""
    probe = os.path.join(BUILD, "_probe.c")
    with open(probe, "w") as fh:
        fh.write("int main(void){return 0;}\n")
    out = os.path.join(BUILD, "_probe.out")
    ok = subprocess.call([cc, *flags, probe, "-o", out],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    for leaf in (probe, out):
        if os.path.exists(leaf):
            os.unlink(leaf)
    return ok


def machine_is_x86():
    return platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")


def build_flags(cc, args):
    """Assembles the compile and link flags for this host and flavour."""
    leaf = os.path.splitext(os.path.basename(cc))[0].lower()
    if leaf == "cl":   # msvc, not clang
        flags = ["/nologo", "/W4", "/std:c11"]
        flags += ["/Od", "/Zi"] if args.debug else ["/O2", "/fp:fast"]
        if not args.no_simd and not args.portable and machine_is_x86():
            flags.append("/arch:AVX2")
        if args.no_simd:
            flags.append("/DILL_NO_SIMD")
        return flags, [], True

    flags = ["-std=c11", "-Wall", "-Wextra"]
    flags += ["-O0", "-g3"] if args.debug else ["-O3", "-fno-math-errno"]

    if args.no_simd:
        flags.append("-DILL_NO_SIMD")
    elif args.portable:
        if machine_is_x86():
            for wanted in (["-mavx2", "-mfma", "-mf16c"], ["-msse4.2"]):
                if accepts(cc, wanted):
                    flags += wanted
                    break
    else:
        for wanted in (["-march=native"], ["-mcpu=native"],
                       ["-mavx2", "-mfma", "-mf16c"]):
            if accepts(cc, wanted):
                flags += wanted
                break

    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]

    link = ["-lm"]
    if os.name != "nt" and accepts(cc, ["-pthread"]):
        flags.append("-pthread")
    return flags, link, False


def compile_one(cc, source, target, flags, link, msvc):
    os.makedirs(BUILD, exist_ok=True)
    if msvc:
        cmd = [cc, *flags, source, f"/Fe:{target}", f"/Fo:{target}.obj"]
    else:
        cmd = [cc, *flags, source, "-o", target, *link]
    return call(cmd, cwd=ROOT)


def do_build(args):
    cc = find_cc(args.cc)
    if not cc:
        print("no C compiler found; set CC or pass --cc")
        return 1
    os.makedirs(BUILD, exist_ok=True)
    flags, link, msvc = build_flags(cc, args)
    say(f"build with {cc}")
    print(f"  flavour: {'no-simd' if args.no_simd else 'portable' if args.portable else 'native'}"
          f"{', debug' if args.debug else ''}{', sanitize' if args.sanitize else ''}")
    suffix = ".exe" if os.name == "nt" else ""
    for source, name in (("app_main.c", "app_main"), ("app_test.c", "app_test")):
        code = compile_one(cc, os.path.join(ROOT, source),
                           os.path.join(BUILD, name + suffix), flags, link, msvc)
        if code != 0:
            print(f"build failed: {source}")
            return code
    return 0


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------

def do_install(args):
    say("install the reference stack")
    if not os.path.exists(VENV):
        if call([sys.executable, "-m", "venv", VENV]) != 0:
            print("could not create .venv; falling back to the current interpreter")
    python = venv_python()
    code = call([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    code |= call([python, "-m", "pip", "install", *NEEDS])
    if code != 0:
        print("\ninstall failed.  The reference stack is only needed by app_test.py;\n"
              "the engine itself builds and runs with no dependencies at all.")
    _ = args
    return 0 if code == 0 else 1


def do_test(args):
    code = do_build(args)
    if code != 0:
        return code
    suffix = ".exe" if os.name == "nt" else ""
    binary = os.path.join(BUILD, "app_main" + suffix)

    say("unit tests")
    code = call([os.path.join(BUILD, "app_test" + suffix)])
    if code != 0:
        return code

    say("reference comparison")
    python = venv_python()
    probe = subprocess.call([python, "-c", "import torch, transformers"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe != 0:
        print("  transformers is not installed; skipping (python3 run.py install)")
        return 0
    cmd = [python, os.path.join(ROOT, "app_test.py"), "--binary", binary]
    if args.model:
        cmd += ["--model", args.model]
    if args.no_checkpoint:
        cmd += ["--no-checkpoint"]
    if args.filter:
        cmd += ["--filter", args.filter]
    return call(cmd)


def do_run(args):
    code = do_build(args)
    if code != 0:
        return code
    suffix = ".exe" if os.name == "nt" else ""
    rest = list(args.rest)
    if not rest:
        rest = ["help"]
    say("run")
    return call([os.path.join(BUILD, "app_main" + suffix), *rest])


def do_bench(args):
    code = do_build(args)
    if code != 0:
        return code
    model = args.model or os.path.join(ROOT, "model")
    if not os.path.isdir(model):
        print(f"bench needs a checkpoint: {model} not found; pass --model PATH")
        return 1
    suffix = ".exe" if os.name == "nt" else ""
    say("bench")
    return call([os.path.join(BUILD, "app_main" + suffix), "bench",
                 "--model", model, *args.rest])


def do_clean(args):
    say("clean")
    shutil.rmtree(BUILD, ignore_errors=True)
    _ = args
    return 0


def do_all(args):
    for step in (do_install, do_build, do_test):
        code = step(args)
        if code != 0 and step is not do_install:
            return code
    return 0


WORKFLOWS = {
    "install": do_install,
    "build": do_build,
    "test": do_test,
    "run": do_run,
    "bench": do_bench,
    "clean": do_clean,
    "all": do_all,
    "ci": do_all,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow", choices=sorted(WORKFLOWS), help="what to do")
    parser.add_argument("--cc", default=None, help="compiler to use")
    parser.add_argument("--portable", action="store_true", help="do not target this host")
    parser.add_argument("--no-simd", action="store_true", help="scalar kernels only")
    parser.add_argument("--debug", action="store_true", help="unoptimised build")
    parser.add_argument("--sanitize", action="store_true", help="address and ub sanitizers")
    parser.add_argument("--model", default=None, help="checkpoint folder")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="skip the checkpoint suite even when ./model exists")
    parser.add_argument("--filter", default=None, help="subset of reference checks to run")
    # Everything after a literal `--` belongs to app_main, not to this script,
    # so it is split off before argparse sees it.
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = parser.parse_args(argv)
    args.rest = rest
    return WORKFLOWS[args.workflow](args)


if __name__ == "__main__":
    sys.exit(main())
