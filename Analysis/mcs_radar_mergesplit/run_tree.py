"""
Run PyFLEXTRKR's run_mcs_radar3d.py from an extracted code tree (e.g. <workdir>/code_old), not from
the editable install, so two code revisions can be run on the same input.

usage: run_tree.py <tree_dir> <config.yml>

An editable install's .pth puts the repo before PYTHONPATH, so PYTHONPATH cannot select the tree; it is put
at sys.path[0] here instead. Dask workers (multiprocessing spawn) inherit sys.path from this process, which
is checked before the run starts.
"""
import sys, os, runpy


def main():
    tree, config = sys.argv[1], sys.argv[2]
    import pyflextrkr
    assert os.path.realpath(pyflextrkr.__file__).startswith(os.path.realpath(tree)), pyflextrkr.__file__
    print("pyflextrkr from:", pyflextrkr.__file__, flush=True)
    if os.environ.get("RUN_TREE_CHECK_WORKERS", "1") == "1":
        from dask.distributed import Client, LocalCluster
        cl = LocalCluster(n_workers=2, threads_per_worker=1)
        c = Client(cl)
        paths = set(c.run(lambda: __import__("pyflextrkr").__file__).values())
        print("workers import pyflextrkr from:", paths, flush=True)
        assert all(os.path.realpath(p).startswith(os.path.realpath(tree)) for p in paths), paths
        c.close()
        cl.close()
    script = os.path.join(tree, "runscripts", "run_mcs_radar3d.py")
    sys.argv = [script, config]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    main()
