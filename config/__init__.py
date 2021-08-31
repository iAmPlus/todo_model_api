import json
import os
import yaml

_config_directory = 'config/'
_setup_file = _config_directory + 'setup.yaml'

def get_env(var_name, default=None):
    """
    :return: the environment variable value parsed as the type of the default
    value
    """
    val = os.getenv(var_name)
    try:
        if not val:
            return default
        if isinstance(default, str):
            return val
        if default is True or default is False:
            return {'true': True, 'false': False}[val.lower()]
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
        if isinstance(default, (dict, list)):
            return json.loads(val)  # TODO Handle booleans in object or list
    except (ValueError, KeyError, json.decoder.JSONDecodeError):
        raise ValueError(
            'Cannot parse environment configuration option %s ' % var_name
            + 'value "%s" (%s) as type %s' % (val, type(val).__name__,
                                              type(default).__name__)
        )
    raise ValueError(
        'Cannot override configuration option %s from environment: ' % var_name
        + 'default value has type %s ' % type(default).__name__
        + 'but I do not know how to parse that.'
    )


def merge_env(settings):
    """
    For every key in a given settings dict, override with env var if exists.
    Allows config to be set in container definitions.
    """
    return {key: get_env(key, value) for key, value in settings.items()}

def read_yaml_from_file(filepath):
    with open(filepath) as fp:
        return yaml.load(fp)

def get_config_filepaths(setup_file=None):
    file_paths = []

    if not setup_file:
        # Use the config file from environmental variable if it is set.
        setup_file = os.environ.get('CLASSIFIER_CONFIG_FILE', setup_file)
    if not setup_file or not os.path.exists(setup_file):
        setup_file = os.path.join('config', 'setup.yaml')

    if os.path.exists(setup_file):
        yaml = read_yaml_from_file(setup_file)

        default_yaml = os.path.join('config',
                                    'default.yaml')
        if os.path.exists(default_yaml):
            file_paths.append(default_yaml)

        file_paths.append(setup_file)
    else:
        assert False, "Unable to determine the config file path"

    return file_paths



def read_config(setup_file=None):
    config = {}
    for filepath in get_config_filepaths(setup_file=setup_file):
        config.update(read_yaml_from_file(filepath))

    config = merge_env(config)
    return config
