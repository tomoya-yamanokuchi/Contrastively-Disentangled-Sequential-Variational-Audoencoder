import slackweb
import datetime
from omegaconf import OmegaConf


class Notifying:
    def __init__(self, path_web_hook_url, text: str=None):
        self.web_hook_url = self.load_web_hook_url(path_web_hook_url)
        if text is None: self.text = "Process is finished [{:}] --> {:} (elapsed_time: {:})"
        else:            self.text = text


    def load_web_hook_url(self, path_web_hook_url):
        yaml = OmegaConf.load(path_web_hook_url)
        assert type(yaml.web_hook_url) is str
        return yaml.web_hook_url


    def elapsed_time_str(self, seconds):
        seconds = int(seconds + 0.5)    # 秒数を四捨五入
        h = seconds // 3600             # 時の取得
        m = (seconds - h * 3600) // 60  # 分の取得
        s = seconds - h * 3600 - m * 60 # 秒の取得
        return f"{h:02}:{m:02}:{s:02}"  # hh:mm:ss形式の文字列で返す


    def notify_slack(self, file_name: str, elapsed_time=None):
        slack               = slackweb.Slack(self.web_hook_url)
        time                = datetime.datetime.now()
        elapsed_time_format = self.elapsed_time_str(elapsed_time)
        text                = "Process is finished [{:}] --> {:} (elapsed_time: {:})"
        slack.notify(text=text.format(time, file_name, elapsed_time_format))


if __name__ == '__main__':
    import time

    notifying = Notifying(
        path_web_hook_url = "/home/tomoya-y/.config/Code/User/slack.yaml"
    )
    notifying.notify_slack(time.time())
