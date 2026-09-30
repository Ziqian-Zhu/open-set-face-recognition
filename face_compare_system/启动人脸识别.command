#!/bin/zsh
set -eu
project_dir="${0:A:h}"
cd "$project_dir"
if [[ ! -x "$project_dir/.venv-ui/bin/python" ]]; then
  print "界面运行环境缺失，请重新配置 .venv-ui（Python + Tk 8.6 或更新版本）。"
  read -k 1 "?按任意键关闭。"
  exit 1
fi
"$project_dir/.venv-ui/bin/python" "$project_dir/main.py" || {
  print "启动失败，请保留上方错误信息。"
  read -k 1 "?按任意键关闭。"
  exit 1
}
