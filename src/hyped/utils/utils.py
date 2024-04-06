import importlib.util


def is_package_installed(package_name):
    """Check if a package is installed

    Arguments:
        package_name (str): name of the package to check for

    Returns:
        is_installed (bool):
            boolean indicating whether the package is installed or not
    """
    return importlib.util.find_spec(package_name) is not None
