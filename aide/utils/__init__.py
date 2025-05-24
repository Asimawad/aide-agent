import logging
import shutil
import zipfile
from pathlib import Path
import json
logger = logging.getLogger("aide")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def extract_archives(path: Path):
    """
    unzips all .zip files within `path` and cleans up task dir

    [TODO] handle nested zips
    """
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            logger.info(
                f"Skipping {zip_f} as an item with the same name already exists."
            )
            # if it's a file, it's probably exactly the same as in the zip -> remove the zip
            # [TODO] maybe add an extra check to see if zip file content matches the colliding file
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
            continue

        logger.info(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)

        # remove any unwanted files
        clean_up_dataset(f_out_dir)

        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.info(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.info(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()


def preproc_data(path: Path):
    extract_archives(path)
    clean_up_dataset(path)


def parse_model_id(model_name: str) -> tuple:
    """
    Parse a model ID into organization and model name components.

    Args:
        model_name: Full model identifier (e.g. "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic")

    Returns:
        Tuple of (organization, model_name) or (None, model_name) if no organization
    """
    if "/" in model_name:
        org, model = model_name.split("/", 1)
        return org, model
    else:
        return None, model_name


def load_benchmarks(competition_name: str):
    json_path = Path(__file__).parent / "competition_template.json"
    with open(json_path, "r") as f:
        benchmarks = json.load(f)
    return benchmarks[competition_name]
