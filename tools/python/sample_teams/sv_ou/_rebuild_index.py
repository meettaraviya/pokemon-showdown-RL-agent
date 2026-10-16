from pathlib import Path

out_dir = Path(__file__).resolve().parent
sources_path = out_dir / "_sources.txt"
index_path = out_dir / "index.tsv"
failed_path = out_dir / "_failed.txt"

links = [line.strip() for line in sources_path.read_text(encoding="utf-8").splitlines() if line.strip()]

lines = ["idx\tteam_file\tpokepaste_url\traw_url\tstatus\tbytes"]
failed = []

for i, url in enumerate(links, start=1):
    team_file = f"team_{i:03d}.txt"
    path = out_dir / team_file
    raw_url = url + "/raw"
    if path.exists():
        size = path.stat().st_size
        status = "ok" if size > 0 else "empty"
    else:
        size = 0
        status = "failed"
        failed.append(f"{i}\t{url}\t{raw_url}\tmissing team file")

    lines.append(f"{i}\t{team_file if status != 'failed' else ''}\t{url}\t{raw_url}\t{status}\t{size}")

index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
if failed:
    failed_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
else:
    if failed_path.exists():
        failed_path.unlink()

print(f"links={len(links)}")
print(f"ok={sum(1 for line in lines[1:] if '\tok\t' in line)}")
print(f"empty={sum(1 for line in lines[1:] if '\tempty\t' in line)}")
print(f"failed={sum(1 for line in lines[1:] if '\tfailed\t' in line)}")
