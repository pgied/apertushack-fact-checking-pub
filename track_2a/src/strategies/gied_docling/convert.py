"""Convert the booklets with docling for gied_docling, and compare how the readers read them.

    uv run python -m src.strategies.gied_docling.convert                          # all booklets of the advanced task, on the CPU
    uv run python -m src.strategies.gied_docling.convert --device mps --limit 3   # a few, on the Mac's GPU
    uv run python -m src.strategies.gied_docling.convert --report                 # only the comparison

docling runs in an environment of its own (`uv run --isolated --with docling==...`), so neither it nor
its models get into our lock file or the Docker image. Each booklet's DoclingDocument is saved as
<out>/<SHA-256 of the PDF>.json, where gied_docling looks for it, and the seconds it took go into
<out>/manifest.jsonl. Booklets already converted are skipped. On Clariden, run it through
slurm/run.sbatch with `--device cuda`.

The report compares the three readers on every converted booklet: seconds per page, and fidelity,
the share of each gold reference text found in what the reader read (dataset.locate).

This file is also the worker that runs inside docling's environment (`--worker`): there it can only
use the standard library and docling.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

DOCLING = "2.138.0"
ROOT = Path(__file__).resolve().parents[3]  # track_2a/


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the booklets with docling and compare the readers.")
    parser.add_argument("--booklets", type=Path, default=ROOT / "data" / "advanced" / "booklets")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "converted" / "docling")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for docling's models")
    parser.add_argument("--limit", type=int, default=0, help="convert only this many booklets")
    parser.add_argument("--report", action="store_true", help="only compare the readers on what is converted")
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return work(args.out, args.device, args.threads, [Path(p) for p in args.worker])

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report:
        todo = [p for p in sorted(args.booklets.glob("*.pdf")) if not (args.out / f"{sha256(p)}.json").exists()]
        todo = todo[: args.limit] if args.limit else todo
        if todo:
            print(f"Converting {len(todo)} booklets with docling {DOCLING} on {args.device} into {args.out}")
            subprocess.run(["uv", "run", "--isolated", "--no-project", "--with", f"docling=={DOCLING}", "python",
                            __file__, "--out", str(args.out), "--device", args.device, "--threads", str(args.threads),
                            "--worker", *map(str, todo)], check=True)
    report(args.booklets, args.out)


def work(out: Path, device: str, threads: int, pdfs: list[Path]) -> None:
    """Inside docling's environment: load docling once, then convert the PDFs one after the other."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # The booklets are born digital: their text layer is exact, so no OCR
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(device), num_threads=threads)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converter.initialize_pipeline(InputFormat.PDF)  # load the models before the clock starts
    for pdf in pdfs:
        begin = time.perf_counter()
        document = converter.convert(pdf).document
        seconds = time.perf_counter() - begin
        digest = sha256(pdf)
        (out / f"{digest}.json").write_text(json.dumps(document.export_to_dict(), ensure_ascii=False), encoding="utf-8")
        line = {"booklet": pdf.name, "sha256": digest, "pages": len(document.pages), "seconds": round(seconds, 2),
                "device": device, "threads": threads, "docling": DOCLING}
        with open(out / "manifest.jsonl", "a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(line) + "\n")
        print(f"  {pdf.name}: {len(document.pages)} pages in {seconds:.1f} s", flush=True)


def report(booklets: Path, out: Path) -> None:
    """Compare the readers on the converted booklets: seconds per page and fidelity."""
    from src import dataset
    from src.strategies._gied import pdf as readers

    manifest = out / "manifest.jsonl"
    converted = {r["booklet"]: r for r in dataset.read_jsonl(manifest)} if manifest.exists() else {}
    if not converted:
        print(f"Nothing converted in {out} yet")
        return
    rows = dataset.read_jsonl(ROOT / "data" / "raw" / dataset.FILENAME)
    references: dict[str, set[str]] = {}
    for row in rows:
        references.setdefault(dataset.booklet_name(row), set()).add(row["reference_string"])

    lines = [f"Readers compared on {len(converted)} booklets converted with docling {DOCLING}", ""]
    totals = {name: [0.0, 0, []] for name in ("pdfium", "tags", "docling")}  # seconds, pages, fidelity shares
    for name, entry in sorted(converted.items()):
        path = booklets / name
        for reader in totals:
            begin = time.perf_counter()
            pages = readers.read(path, reader, out)
            seconds = entry["seconds"] if reader == "docling" else time.perf_counter() - begin
            texts = [" ".join(block.text for block in page) for page in pages]
            shares = [dataset.locate(reference, texts)[1] for reference in references.get(name, ())]
            totals[reader][0] += seconds
            totals[reader][1] += len(pages)
            totals[reader][2] += shares
    lines.append(f"{'reader':<8} {'s/page':>8} {'fidelity mean':>14} {'min':>6} {'≥ 90 %':>7}")
    for reader, (seconds, pages, shares) in totals.items():
        if shares:
            lines.append(f"{reader:<8} {seconds / max(1, pages):>8.3f} {sum(shares) / len(shares):>14.1%} "
                         f"{min(shares):>6.0%} {sum(s >= 0.9 for s in shares) / len(shares):>7.0%}")
    lines += ["", "s/page: docling on " + ", ".join(sorted({e['device'] for e in converted.values()})) +
              " as converted; pdfium and tags measured now. Fidelity: the share of each gold reference text "
              "found in the reader's text (dataset.locate), over the distinct references of these booklets."]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
