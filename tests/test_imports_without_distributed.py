"""
The modules of the radar_feature MCS workflow and of trackpcp import without the `distributed` package: the CI environment (requirements-ci.txt and
`pip install -e .`) has `dask` but not `distributed`, and a test module that imports a module with `from dask.distributed import ...` at the top
cannot even be collected there (it stops the whole run). `wait` of dask.distributed is imported in the parallel branch, where it is used.
The check runs in a new interpreter in which `import distributed` fails, like in an environment without the package.
"""
import subprocess
import sys

MODULES = [
    'pyflextrkr.regionprops_driver',
    'pyflextrkr.regionprops_func',
    'pyflextrkr.mapfeature_driver',
    'pyflextrkr.movement_speed',
    'pyflextrkr.trackpcp_driver',
    'pyflextrkr.trackpcp_func',
    'pyflextrkr.parallel_utils',
    'pyflextrkr.identifymcs_radar',
    'pyflextrkr.mcs_lifecycle',
    'pyflextrkr.mapfeature_func',
    'pyflextrkr.trackstats_func',
]

CODE = """
import importlib, sys
sys.modules['distributed'] = None      # `import distributed` raises ImportError, like a package that is not installed
failed = []
for name in %r:
    try:
        importlib.import_module(name)
    except ImportError as error:
        failed.append(f'{name}: {error}'.splitlines()[0])
print('\\n'.join(failed))
sys.exit(1 if failed else 0)
""" % (MODULES,)


def test_the_radar_and_trackpcp_modules_import_without_the_distributed_package():
    result = subprocess.run([sys.executable, '-c', CODE], capture_output=True, text=True)
    assert result.returncode == 0, f'modules that need the distributed package at import:\n{result.stdout}{result.stderr}'
