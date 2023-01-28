from socket import gethostname

def get_pc_name():
    hostname_dict = {
        "dl-box0"              : "dl-box",
        "G-Master-Spear-MELCO" : "melco",
        "tomoya-y-device"      : "remote_3090",
        "tsukumo3090ti"        : "remote_tsukumo3090ti",
    }
    name = gethostname()
    return hostname_dict[name]