import datetime
import glob
import json
import os
import re
import shutil
from distutils.version import LooseVersion
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from pathlib import Path
from pprint import pprint
from types import ModuleType
from typing import List, Optional, Sequence
from urllib import request
from urllib.request import Request, urlopen

import jsonargparse
import pkg_resources
from packaging.version import parse as version_parse

REQUIREMENT_FILES = {
    "pytorch": (
        "requirements/pytorch/base.txt",
        "requirements/pytorch/extra.txt",
        "requirements/pytorch/strategies.txt",
        "requirements/pytorch/examples.txt",
    ),
    "app": (
        "requirements/app/base.txt",
        "requirements/app/ui.txt",
        "requirements/app/cloud.txt",
    ),
    "lite": (
        "requirements/lite/base.txt",
        "requirements/lite/strategies.txt",
    ),
}
REQUIREMENT_FILES_ALL = list(chain(*REQUIREMENT_FILES.values()))
PACKAGE_MAPPING = {"app": "lightning-app", "pytorch": "pytorch-lightning"}


def pypi_versions(package_name: str, drop_pre: bool = True) -> List[str]:
    """Return a list of released versions of a provided pypi name.

    >>> _ = pypi_versions("lightning_app", drop_pre=False)
    """
    # https://stackoverflow.com/a/27239645/4521646
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = json.load(urlopen(Request(url)))
    versions = list(data["releases"].keys())
    # todo: drop this line after cleaning Pypi history from invalid versions
    versions = list(filter(lambda v: v.count(".") == 2, versions))
    if drop_pre:
        versions = list(filter(lambda v: all(c not in v for c in ["rc", "dev"]), versions))
    versions.sort(key=version_parse)
    return versions


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _retrieve_files(directory: str, *ext: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(os.path.join(root, fname))

    return all_files


class AssistantCLI:
    _PATH_ROOT = str(Path(__file__).parent.parent)
    _PATH_SRC = os.path.join(_PATH_ROOT, "src")

    @staticmethod
    def prepare_nightly_version(proj_root: str = _PATH_ROOT) -> None:
        """Replace semantic version by date."""
        path_info = os.path.join(proj_root, "pytorch_lightning", "__about__.py")
        # get today date
        now = datetime.datetime.now()
        now_date = now.strftime("%Y%m%d")

        print(f"prepare init '{path_info}' - replace version by {now_date}")
        with open(path_info, encoding="utf-8") as fp:
            init = fp.read()
        init = re.sub(r'__version__ = [\d\.\w\'"]+', f'__version__ = "{now_date}"', init)
        with open(path_info, "w", encoding="utf-8") as fp:
            fp.write(init)

    @staticmethod
    def requirements_prune_pkgs(packages: Sequence[str], req_files: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Remove some packages from given requirement files."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for req in req_files:
            AssistantCLI._prune_packages(req, packages)

    @staticmethod
    def _prune_packages(req_file: str, packages: Sequence[str]) -> None:
        """Remove some packages from given requirement files."""
        path = Path(req_file)
        assert path.exists()
        text = path.read_text()
        lines = text.splitlines()
        final = []
        for line in lines:
            ln_ = line.strip()
            if not ln_ or ln_.startswith("#"):
                final.append(line)
                continue
            req = list(pkg_resources.parse_requirements(ln_))[0]
            if req.name not in packages:
                final.append(line)
        pprint(final)
        path.write_text("\n".join(final))

    @staticmethod
    def _replace_min(fname: str) -> None:
        req = open(fname, encoding="utf-8").read().replace(">=", "==")
        open(fname, "w", encoding="utf-8").write(req)

    @staticmethod
    def replace_oldest_ver(requirement_fnames: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Replace the min package version by fixed one."""
        for fname in requirement_fnames:
            AssistantCLI._replace_min(fname)

    @staticmethod
    def _release_pkg(pkg: str, src_folder: str = _PATH_SRC) -> bool:
        pypi_ver = pypi_versions(pkg)[-1]
        _version = _load_py_module("version", os.path.join(src_folder, pkg.replace("-", "_"), "__version__.py"))
        local_ver = _version.version
        return "dev" not in local_ver and LooseVersion(local_ver) > LooseVersion(pypi_ver)

    @staticmethod
    def determine_releasing_pkgs(
        src_folder: str = _PATH_SRC, packages: Sequence[str] = ("pytorch", "app"), inverse: bool = False
    ) -> Sequence[str]:
        """Determine version of package where the name is `lightning.<name>`."""
        if isinstance(packages, str):
            packages = [packages]
        releasing = [pkg for pkg in packages if AssistantCLI._release_pkg(PACKAGE_MAPPING[pkg], src_folder=src_folder)]
        if inverse:
            releasing = list(filter(lambda pkg: pkg not in releasing, packages))
        return json.dumps([{"pkg": pkg for pkg in releasing}])

    @staticmethod
    def download_package(package: str, folder: str = ".", version: Optional[str] = None) -> None:
        """Download specific or latest package from PyPI where the name is `lightning.<name>`."""
        url = f"https://pypi.org/pypi/{PACKAGE_MAPPING[package]}/json"
        data = json.load(urlopen(Request(url)))
        if not version:
            pypi_vers = pypi_versions(PACKAGE_MAPPING[package], drop_pre=False)
            version = pypi_vers[-1]
        releases = list(filter(lambda r: r["packagetype"] == "sdist", data["releases"][version]))
        assert releases, f"Missing 'sdist' for this package/version aka {package}/{version}"
        release = releases[0]
        pkg_url = release["url"]
        pkg_file = os.path.basename(pkg_url)
        pkg_path = os.path.join(folder, pkg_file)
        os.makedirs(folder, exist_ok=True)
        print(f"downloading: {pkg_url}")
        request.urlretrieve(pkg_url, pkg_path)

    @staticmethod
    def _find_pkgs(folder: str, pkg_pattern: str = "lightning") -> List[str]:
        """Find all python packages with spec.

        pattern in given folder, in case `src` exists dive there.
        """
        pkg_dirs = [d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)]
        if "src" in [os.path.basename(p) for p in pkg_dirs]:
            return AssistantCLI._find_pkgs(os.path.join(folder, "src"), pkg_pattern)
        pkg_dirs = list(filter(lambda p: pkg_pattern in os.path.basename(p), pkg_dirs))
        return pkg_dirs

    @staticmethod
    def mirror_pkg2source(pypi_folder: str, src_folder: str) -> None:
        """From extracted sdist packages overwrite the python package with given pkg pattern."""
        pypi_dirs = [d for d in glob.glob(os.path.join(pypi_folder, "*")) if os.path.isdir(d)]
        for pkg_dir in pypi_dirs:
            for py_dir in AssistantCLI._find_pkgs(pkg_dir):
                dir_name = os.path.basename(py_dir)
                py_dir2 = os.path.join(src_folder, dir_name)
                shutil.rmtree(py_dir2, ignore_errors=True)
                shutil.copytree(py_dir, py_dir2)

    @staticmethod
    def copy_replace_imports(
        source_dir: str, source_import: str, target_import: str, target_dir: Optional[str] = None
    ) -> None:
        """Recursively replace imports in given folder."""

        source_imports = source_import.strip().split(",")
        target_imports = target_import.strip().split(",")
        assert len(source_imports) == len(target_imports), (
            "source and target imports must have the same length, "
            f"source: {len(source_import)}, target: {len(target_import)}"
        )

        if target_dir is None:
            target_dir = source_dir

        ls = _retrieve_files(source_dir)

        for fp in ls:
            if fp.endswith(".py"):
                with open(fp, encoding="utf-8") as fo:
                    py = fo.readlines()

                for source_import, target_import in zip(source_imports, target_imports):
                    for i, ln in enumerate(py):
                        py[i] = re.sub(rf"([^_]|^){source_import}([^_\w]|$)", rf"\1{target_import}\2", ln)

                if target_dir:
                    fp_new = fp.replace(source_dir, target_dir)
                    os.makedirs(os.path.dirname(fp_new), exist_ok=True)
                else:
                    fp_new = fp

                with open(fp_new, "w", encoding="utf-8") as fo:
                    fo.writelines(py)
            elif not fp.endswith(".pyc"):
                fp_new = fp.replace(source_dir, target_dir)
                os.makedirs(os.path.dirname(fp_new), exist_ok=True)
                if os.path.abspath(fp) != os.path.abspath(fp_new):
                    shutil.copy2(fp, fp_new)


if __name__ == "__main__":
    jsonargparse.CLI(AssistantCLI, as_positional=False)
