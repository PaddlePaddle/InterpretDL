

def versiontuple2tuple(v):
    "ref: https://stackoverflow.com/a/11887825/4834515"
    return tuple(map(int, (v.split("."))))
