def turn_on_video_recording():
    import builtins
    builtins.visualize = True


def turn_off_video_recording():
    import builtins
    builtins.visualize = False


def get_video_recording_status():
    import builtins
    return getattr(builtins, "visualize", False)
