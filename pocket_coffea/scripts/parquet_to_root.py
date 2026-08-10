import glob
import os

import awkward as ak
import click
import uproot


def _without_cycle(path):
    return path.split(";", 1)[0]


def _tree_output_path(input_dir, root_file, tree_path):
    relative_dir = os.path.relpath(os.path.dirname(root_file), input_dir)
    parts = [] if relative_dir == "." else relative_dir.split(os.path.sep)
    parts.extend(_without_cycle(tree_path).split("/"))
    return "/".join(part for part in parts if part)


def _is_tree(classname):
    return classname.startswith("TTree") or classname == "ROOT::RNTuple"


def merge_root_directory(input_dir, output_file):
    root_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.root"), recursive=True))
    output_abs = os.path.abspath(output_file)
    arrays_by_tree = {}

    for root_file in root_files:
        if os.path.abspath(root_file) == output_abs or os.path.getsize(root_file) == 0:
            continue
        with uproot.open(root_file) as handle:
            for tree_path, classname in handle.classnames().items():
                if not _is_tree(classname):
                    continue
                output_tree = _tree_output_path(input_dir, root_file, tree_path)
                payload = handle[tree_path].arrays(library="ak")
                if output_tree in arrays_by_tree:
                    arrays_by_tree[output_tree].append(payload)
                else:
                    arrays_by_tree[output_tree] = [payload]

    with uproot.recreate(output_file) as output:
        for tree_path, chunks in arrays_by_tree.items():
            if len(chunks) == 1:
                output[tree_path] = chunks[0]
            else:
                output[tree_path] = ak.concatenate(chunks)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_file", type=click.Path(dir_okay=False))
def main(input_dir, output_file):
    merge_root_directory(input_dir, output_file)


if __name__ == "__main__":
    main()