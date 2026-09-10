# Environment tuning — keeping this machine usable

Measured 2026-09-10. This project handles ~200,000 dataset files on a machine with **7.8 GB
of RAM**, and the symptoms of exhaustion are not obviously memory-related: Pylance dies, the
integrated terminal crashes, git fails with `Not enough memory resources are available to
complete this operation`, and the IDE language-server connection drops.

Three fixes, in order of impact.

---

## 1. Move the pagefile to E: — **needs admin and a reboot**

The problem, as measured:

| Drive | Free | Used |
|---|---|---|
| **C:** | **16.6 GB** | 459.3 GB |
| **E:** | **469.6 GB** | 1393.4 GB |

```
Get-CimInstance Win32_PageFileUsage
Name            AllocatedBaseSize CurrentUsage PeakUsage
C:\pagefile.sys             24576        21057     21108
```

A **24 GB pagefile, 21 GB of it in use, on a drive with 16.6 GB free** — and
`AutomaticManagedPagefile = True`, so Windows sizes it itself and cannot grow it any
further. With 7.8 GB of RAM, this machine leans on the pagefile constantly, and when the
pagefile cannot expand, unrelated allocations start failing. That is the crash.

E: has 469 GB free and is already where every dataset, the virtualenv and all generated
output live. The pagefile belongs there too.

### Do it in the GUI (recommended — clearest feedback)

1. Win+R → `sysdm.cpl` → **Advanced** → Performance **Settings…** → **Advanced** →
   Virtual memory **Change…**
2. Untick **Automatically manage paging file size for all drives**.
3. Select **C:** → **No paging file** → **Set**.
   *(If Windows objects to having no pagefile on the system drive, give C: a small
   fixed one instead — Initial 1024 MB, Maximum 2048 MB — which keeps crash dumps working.)*
4. Select **E:** → **Custom size** → Initial **16384** MB, Maximum **32768** MB → **Set**.
5. **OK**, then **reboot**. The change does not take effect until you do.

### Or from an elevated PowerShell

Run as Administrator. Reboot afterwards.

```powershell
# stop Windows managing it
$cs = Get-CimInstance Win32_ComputerSystem
Set-CimInstance $cs -Property @{ AutomaticManagedPagefile = $false }

# small fixed pagefile on C: so crash dumps still work
$c = Get-CimInstance Win32_PageFileSetting -Filter "Name='C:\\pagefile.sys'"
if ($c) { Set-CimInstance $c -Property @{ InitialSize = 1024; MaximumSize = 2048 } }

# the real one on E:
New-CimInstance -ClassName Win32_PageFileSetting `
  -Property @{ Name = 'E:\pagefile.sys'; InitialSize = 16384; MaximumSize = 32768 }

Restart-Computer
```

Verify after reboot:

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, PeakUsage
```

Both entries should appear, with the E: one carrying the load.

---

## 2. Keep the editor out of `data/` — **already done**

`.vscode/settings.json` excludes `data/`, `.venv/`, `vendor/` and `tools/bin/` from the file
watcher, Pylance analysis and search, and disables git autorefresh (git's status walk is slow
with 200,000 ignored files).

**Reload the VS Code window** after any change here — the settings are read at window start.
Command Palette → *Developer: Reload Window*.

Do not remove those excludes. Each watched directory costs a handle and memory, and `data/`
contains tens of thousands of directories that never need watching.

That file also sets `python.analysis.extraPaths: ["src"]`, which fixes a Pylance error that
is real rather than memory-driven: the package lives in `src/fpe` and scripts add it to
`sys.path` at runtime, so without it every `from fpe...` import is flagged unresolved.

---

## 3. Keep temp files off C: — **done 2026-09-10**

Policy for this project: **write to E: unless something compels C:.** C: carries the
pagefile and has the least room; E: has ~470 GB and already holds the datasets, the
virtualenv and every generated corpus.

Applied as User-scope environment variables (no admin needed; effective in new processes):

| Variable | Value |
|---|---|
| `TEMP` | `E:\localtemp` |
| `TMP` | `E:\localtemp` |
| `PIP_CACHE_DIR` | `E:\localcache\pip` |

Both targets sit outside the repo, so temp files never show up in `git status`.

Also reclaimed: `python -m pip cache purge` removed **4.4 GB** (1,190 files) of wheel cache
from the pip directory under `AppData\Local`. C: went from 16.5 GB to **20.6 GB** free. That
cache is purely re-downloadable, and future pip runs write to E: instead.

To revert:

```powershell
[Environment]::SetEnvironmentVariable('TEMP', "$env:LOCALAPPDATA\Temp", 'User')
[Environment]::SetEnvironmentVariable('TMP',  "$env:LOCALAPPDATA\Temp", 'User')
[Environment]::SetEnvironmentVariable('PIP_CACHE_DIR', $null, 'User')
```

### Still on C:, and worth knowing

| Item | Size | Note |
|---|---|---|
| `Downloads` | **96 GB** | **The single biggest consumer on the drive.** Personal data, so untouched. Windows can relocate it properly: right-click Downloads → Properties → **Location** → Move… → an E: path. That moves the contents and repoints every app that writes there. |
| `AppData\Local\CrashDumps` | 0.18 GB | Safe to clear by hand; a tooling guard blocks automated deletion under `AppData`. |
| the IDE scratchpad | small | Lives under the old `TEMP`. New sessions use `E:\localtemp` now the variable is set. |

Moving Downloads plus the pagefile would take C: from ~20 GB free to well over 100 GB, at
which point none of this is a constraint any more.

---

## While working — operational rules

These are in `PROJECT_RULES.md` as non-negotiable 6, repeated here with the reasoning:

- **Cap generation at 4 threads** while the editor is open. An 8-thread Anguli run is what
  drove the pagefile to its 21 GB peak and took down the terminal, the language server and
  git's credential helper in one go.
- **Dataloader workers 0–2.** Each worker forks a copy of the dataset object.
- **Stream, never bulk-load.** No `np.stack` over a whole corpus; the manifest CSVs exist so
  the file list can be held without the images.
- **Long runs: close the editor**, or run them when you are away from the machine. Expect
  interrupted runs — `scripts/reconcile_anguli.py` exists because one already happened, and
  it repairs the torn records an interrupted generation leaves behind.

## Diagnosing it again

If the terminal or Pylance starts crashing, check these before anything else:

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object @{n='FreeGB';e={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage
Get-PSDrive C, E | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 `
  @{n='MB';e={[math]::Round($_.WorkingSet64/1MB)}}, ProcessName
```

A `PeakUsage` close to `AllocatedBaseSize` means the pagefile is the binding constraint, not
any single process.
