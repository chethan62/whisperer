# Licence texts shipped with the prebuilt binaries

These are the licence **texts** that have to travel with a redistribution of the code the prebuilt binaries
bundle. They are not transcripts of the project's own licence — that is [`../LICENSE`](../LICENSE) (MIT).

| file | covers | source |
|---|---|---|
| `LGPL-3.0.txt` + `GPL-3.0.txt` | PySide6 / Qt, which the prebuilt binaries bundle | [gnu.org/licenses/lgpl-3.0.txt](https://www.gnu.org/licenses/lgpl-3.0.txt) · [gpl-3.0.txt](https://www.gnu.org/licenses/gpl-3.0.txt) |
| `Apache-2.0.txt` | `tokenizers`, `huggingface_hub`, `hf-xet`, `packaging` | [apache.org/licenses/LICENSE-2.0.txt](https://www.apache.org/licenses/LICENSE-2.0.txt) |

LGPL-3.0 is a supplement to GPL-3.0 and its section 0 says the licence consists of both documents, which is why
both are here. `build.py` adds them to the bundle together with a generated `INDEX.txt` and one notice file per
bundled distribution (name, version, declared licence — taken from the wheel's own metadata, which is the only
licence statement most wheels carry; they ship no text at all, PySide6 included).

## Copyleft among the bundled components

Two of them are not permissive, and each has an obligation beyond attribution:

- **PySide6 / Qt — LGPL-3.0** (or GPL-2.0/3.0). The binaries are built `--onedir`, so the Qt libraries sit in
  the same folder as the executable and stay replaceable; the licence text ships here. Qt is used unmodified.
- **tqdm — MPL-2.0 AND MIT.** MPL-2.0 §3.2 asks a distributor of executable form to say how the source can be
  obtained: it is unmodified, and available from [pypi.org/project/tqdm](https://pypi.org/project/tqdm/) (sdist)
  or [github.com/tqdm/tqdm](https://github.com/tqdm/tqdm).

Everything else is MIT / BSD / Apache-2.0. The authoritative per-build list is the `INDEX.txt` this folder's
generator writes into each release, not a list transcribed here.

Hashes of the shipped texts (they are licence documents: a byte changed is a different document) —

```
e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118  LGPL-3.0.txt
3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986  GPL-3.0.txt
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30  Apache-2.0.txt
```

The Python wheels themselves are **not** redistributed by this repository — they are installed from PyPI at
build time — so these texts exist for the artefacts the project does publish: the binaries in `releases`.
