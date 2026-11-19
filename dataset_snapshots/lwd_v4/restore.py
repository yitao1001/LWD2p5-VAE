"""Restore the published v4 dataset and check each file against its source SHA-256."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile


SNAPSHOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SNAPSHOT.parents[1] / "datasets" / "lwd_v4"


def restore(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    checksums = {}
    for line in (SNAPSHOT / "files.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid snapshot path: {name}")
        checksums[name] = digest

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-v4-", dir=output.parent) as temporary:
        work = Path(temporary)
        archive = work / "lwd_v4.tar.xz"
        with archive.open("wb") as combined:
            for part in sorted(SNAPSHOT.glob("lwd_v4.tar.xz.[0-9][0-9][0-9]")):
                with part.open("rb") as source:
                    shutil.copyfileobj(source, combined, length=1024 * 1024)
        data = work / "verified"
        data.mkdir()
        with tarfile.open(archive, mode="r|xz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in checksums:
                    raise ValueError(f"Unexpected archive entry: {member.name}")
                target = data / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with tar.extractfile(member) as source, target.open("xb") as dest:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        dest.write(chunk)
                if digest.hexdigest() != checksums.pop(member.name):
                    raise ValueError(f"SHA-256 mismatch: {member.name}")
                print(f"Verified {member.name}", flush=True)
        if checksums:
            raise ValueError(f"Missing files: {', '.join(checksums)}")
        data.rename(output)
    print(f"Restored dataset: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    restore(parser.parse_args().output.expanduser())
