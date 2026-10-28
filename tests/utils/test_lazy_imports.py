import ast
import importlib
import inspect
import pathlib
import subprocess
import sys

import pytest

LAZY_MODULES = ["dtaianomaly.anomaly_detection", "dtaianomaly.evaluation"]
HEAVY_DEPENDENCIES = ["torch", "statsmodels", "sktime", "stumpy"]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _type_checking_imports(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {}
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    source = "." * statement.level + (statement.module or "")
                    for alias in statement.names:
                        imports[alias.name] = source
    return imports


@pytest.fixture(params=LAZY_MODULES)
def module(request):
    return importlib.import_module(request.param)


class TestLazyImportsDeclaration:

    def test_lazy_imports_match_all(self, module):
        assert set(module._LAZY_IMPORTS) == set(module.__all__)

    def test_all_has_no_duplicates(self, module):
        assert len(module.__all__) == len(set(module.__all__))


class TestLazyImportsGetattr:

    def test_every_public_name_resolves(self, module):
        for name in module.__all__:
            assert getattr(module, name) is not None

    def test_object_comes_from_declared_submodule(self, module):
        for name in module.__all__:
            submodule = importlib.import_module(
                module._LAZY_IMPORTS[name], module.__name__
            )
            assert getattr(module, name) is getattr(submodule, name)

    def test_unknown_attribute_raises(self, module):
        with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
            getattr(module, "DoesNotExist")


class TestLazyImportsTypeChecking:

    def test_type_checking_names_match_all(self, module):
        assert set(_type_checking_imports(module)) == set(module.__all__)

    def test_type_checking_modules_match_lazy_imports(self, module):
        assert _type_checking_imports(module) == module._LAZY_IMPORTS


class TestLazyImportsDir:

    def test_all_names_in_dir(self, module):
        assert set(module.__all__) <= set(dir(module))


class TestLazyImportsNoHeavyDependencies:

    def test_import_does_not_load_heavy_dependencies(self):
        code = (
            "import sys\n"
            "import dtaianomaly.anomaly_detection, dtaianomaly.evaluation, dtaianomaly.data\n"
            f"print(','.join(m for m in {HEAVY_DEPENDENCIES!r} if m in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == ""
