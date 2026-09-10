# Building NBIS on this machine

NBIS 5.0.0 is from 2015 and its Windows instructions target MinGW/MSYS with
GCC 4.5. Building it with a 2026 toolchain needs four patches, none of them
obvious. This records them so the build is reproducible and so nobody spends
an afternoon rediscovering them.

`mindtct` and `bozorth3` produce the thesis's headline number, so this is the
critical path — see `docs/brainstorms/2026-09-11-wear-aware-abstaining-enhancement-requirements.md`.

## What we ended up with

| | |
|---|---|
| Compiler | winlibs MinGW-w64, **GCC 16.2.0** UCRT, portable zip, extracted to `E:\toolchains\mingw64` |
| CMake | `pip install cmake==3.31.6` into the project venv |
| Make | `mingw32-make.exe` copied to `E:\toolchains\bin\make.exe` |
| Source | `https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip` (52.6 MB), extracted to `E:\toolchains\Rel_5.0.0` |
| Install prefix | `E:\toolchains\nbis` |
| Staged into repo | `tools/bin/` — `mindtct`, `bozorth3`, `cjpegl`, `djpegl`, `cwsq`, `dwsq` (gitignored) |

Nothing goes on C:. There is no compiler, CMake or MSYS installed system-wide;
everything lives under `E:\toolchains` and is reachable by putting
`E:/toolchains/bin`, the venv `Scripts`, and `E:/toolchains/mingw64/bin` on
PATH in that order. The binaries link against the system UCRT and run without
any MinGW DLL on PATH, which is why staging them into `tools/bin/` works.

## The four patches

### 1. CMake must be 3.31, not 4.x

NBIS declares `cmake_minimum_required(VERSION 2.6)`. CMake 4.0 **removed**
compatibility with anything below 3.5 and errors out. `pip install cmake`
gives 4.4 by default, so pin it:

    pip install "cmake==3.31.6"

### 2. `setup.sh` must emit Windows paths, not Git Bash paths

`setup.sh` sets `MAIN_DIR=$PWD`, which under Git Bash is `/e/toolchains/...`.
Native `mingw32-make` cannot resolve that, and every generated makefile fails
with `No rule to make target '/e/toolchains/Rel_5.0.0/rules.mak'`.

Patch the `--MSYS` branch to use Git Bash's Windows-form `pwd -W`:

```sh
if [ $MSYS_FLAG -eq 1 ]; then
        MAIN_DIR=`pwd -W`
        cd ..
        NBIS_DIR=`pwd -W`
```

`DIR_ROOT` in the generated `rules.mak` should then read `E:/toolchains/Rel_5.0.0`.

### 3. `-fpermissive -fcommon` in CFLAGS

Two separate GCC default changes break 1990s C:

- **GCC 14** turned implicit function declarations, int-conversion and
  incompatible-pointer-types into hard **errors**. `-fpermissive` downgrades
  them back to warnings, which `-w` then suppresses.
- **GCC 10** made `-fno-common` the default, so a tentative definition in a
  header included by two translation units becomes
  `multiple definition of 'histo_head'` at link time. `-fcommon` restores the
  old behaviour.

Both go into the `CFLAGS` line of the generated `rules.mak`:

    CFLAGS := -O2 -w -ansi -fpermissive -fcommon -DOPJ_STATIC ...

Apply after every `setup.sh` run — `setup.sh` regenerates `rules.mak`.

### 4. The dependency-file generator mis-parses drive letters

`buildutil/{bin,lib,pcasysx_lib}.mak` define a `make-depend` macro whose second
`sed` pass adds phony targets for headers. It strips everything up to the first
colon with `s/^[^:]*: *//` — and on `E:/toolchains/...` the first colon is the
drive letter's. The result is a mangled second line in every `.d` file:

    E:/.../isamax.o E:/.../isamax.d: isamax.c E:/.../f2c.h
    /toolchains/.../isamax.o E:/.../isamax.d: isamax.c E:/.../f2c.h :

which make rejects with `*** multiple target patterns. Stop.`

Delete the second sed pass entirely, leaving:

```make
define make-depend
	@$(CC) $(CFLAGS) -I$(DIR_INC) $(EXT_INCS) $(M) $1 | \
	$(SED) 's,^$4 *:, $2 $3:,' > $3.tmp
	@$(MV) $3.tmp $3
endef
```

Real dependency tracking is preserved; only the phony-header trick is lost,
which matters solely if a header is deleted mid-build. Delete all existing
`.d` files after patching, or the mangled ones are still read.

## Build sequence

```sh
export PATH="/e/toolchains/bin:/e/biometric-enhancement/.venv/Scripts:/e/toolchains/mingw64/bin:$PATH"
cd /e/toolchains/Rel_5.0.0
mkdir -p /e/toolchains/nbis                    # setup.sh refuses a missing target
./setup.sh E:/toolchains/nbis --MSYS --64 --STDLIBS
sed -i 's/^CFLAGS\t\t:= -O2 -w -ansi/CFLAGS\t\t:= -O2 -w -ansi -fpermissive -fcommon/' rules.mak
# apply patch 4 to buildutil/*.mak, then:
find . -name "*.d" -delete
make config && make it && make install
```

`--STDLIBS` skips the bundled OpenJPEG and libpng, the two third-party
components most likely to fail against GCC 16. The cost is that **mindtct
cannot read PNG** in this build; see below.

## Image format: why lossless JPEG

`mindtct` reads ANSI/NIST, WSQ, lossless JPEG, JPEG and IHead. It does **not**
read TIFF, BMP or PNG — which is every format our datasets ship in — and it
cannot infer the dimensions of a headerless raw file (`image type UNKNOWN`).

So each image is converted before extraction. Of the formats available:

- **WSQ** is the FBI standard and `cwsq` is right there, but it is lossy.
  Putting lossy compression underneath every number in the thesis is not
  acceptable when a lossless option exists.
- **Lossless JPEG** via `cjpegl` is lossless and NBIS-native.

Verified rather than assumed: a 640×480 FVC2004 image round-tripped through
`cjpegl` → `djpegl -raw_out` came back byte-identical, all 307,200 bytes.
`src/fpe/data/convert.py` implements this path; the PNG copy it also writes is
what NFIQ 2 scores and what the network will consume, and has identical pixels.

Two CLI quirks worth knowing: `cjpegl` writes its output beside the *input*
file and leaves a `.ncm` comment file, and `djpegl` needs `-raw_out` to emit
raw bytes — without it the output file is zero length, and it segfaults on
some output extensions.

## Verification

The Phase 0 gate, both rows in `results/registry/runs.jsonl`:

| Dataset | EER | 95% CI | NFIQ 2 mean | FTA |
|---|---|---|---|---|
| FVC2004 DB1_B (deliberately dry / distorted) | 0.1142 | [0.089, 0.141] | 56.0 | 0% |
| Neurotech CrossMatch (clean optical) | 0.0066 | [0.004, 0.011] | 73.4 | 0% |

The 17× gap in the expected direction, with NFIQ 2 tracking it, is the evidence
that the harness measures what it claims to.
