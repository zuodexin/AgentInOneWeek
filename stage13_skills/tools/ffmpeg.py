import subprocess


def run_ffmpeg(ffmpeg_options: str, timeout: int = 10):
    """
    使用 ffmpeg
    Args:
        ffmpeg_options: strings after ffmpeg, e.g. `-i input.mp4 -vn -acodec copy output.aac`
        timeout: maximum time in seconds to allow for the ffmpeg process
    """

    command = f"ffmpeg {ffmpeg_options}"
    subprocess.run(command, shell=True, check=True, timeout=timeout)
