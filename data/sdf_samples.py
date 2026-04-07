import os


def resolve_samples_npz(sdf_data_dir: str, label: str) -> str | None:
    """
    Find samples.npz for a potato label.

    Tries, in order (nested laser/ layout first, then flat layout):
        <sdf_data_dir>/<label>/laser/samples.npz
        <sdf_data_dir>/<label>/samples.npz
    """
    candidates = (
        os.path.join(sdf_data_dir, label, 'laser', 'samples.npz'),
        os.path.join(sdf_data_dir, label, 'samples.npz'),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
