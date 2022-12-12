import slackweb
web_hook_url = 'https://hooks.slack.com/services/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'
slack = slackweb.Slack(web_hook_url)
slack.notify(text='Python投稿テスト')
slack.notify(text='たんたんとした')
slack.notify(text='愛がわからなくなって')
slack.notify(text='すぐに地図をみてる')
slack.notify(text='いくつも灯台があるね')