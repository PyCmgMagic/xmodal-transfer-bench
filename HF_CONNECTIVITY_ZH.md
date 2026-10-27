# HF 登录连接排查和单命令代理

## 本次故障

2026-10-27，原服务器的 hf auth login 在访问官方 /api/whoami-v2 时连接失败，尚未验证 token。
只读检查发现：没有配置 HTTP/HTTPS/ALL_PROXY，也没有 IPv6 默认路由；系统 DNS 与 HTTPS DNS 服务对 HF 域名的查询结果不一致。
使用可信查询获得的 IPv4 地址并保留 TLS 主机名验证时，连接仍被重置。当前直连不可用，不是增加登录次数可以解决的问题。

## 从自己的电脑建立转发

以下命令在你自己的电脑终端运行，不是在服务器终端。连接对象是原项目所在服务器，不是 KIT 集群。
把 SERVER_SSH_ALIAS 换成平时 SSH/VS Code 连接这台服务器时使用的别名，也可写成 用户名@服务器地址；如原 SSH 命令用了非默认端口、跳板或密钥，保留相应选项。

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 127.0.0.1:17890 SERVER_SSH_ALIAS
```

这是 SSH 的远程动态转发（电脑 SSH 客户端需支持该语法）。它在服务器 127.0.0.1:17890 监听 SOCKS 端口，由你电脑上的 SSH 客户端连接目标网站。
只绑定服务器回环地址，不向公网开放代理。保持该电脑终端开启。服务器必须允许 SSH TCP 转发，端口也必须空闲。
本机服务器安装的 OpenSSH 手册已核对 -R 不带目标主机时的 SOCKS 4/5 行为；实际可用性仍需建立转发后验证。

注意：电脑浏览器可访问 HF 不一定表示电脑终端可直接访问。若电脑仅通过 Clash 等代理软件访问 HF，应转发代理软件的实际 HTTP/mixed 端口，而不是使用上述远程动态转发。当前用户已确认 Clash 的 HTTP/mixed 端口为 7897，对应命令：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 127.0.0.1:17890:127.0.0.1:7897 SERVER_SSH_ALIAS
```

此方式的服务器代理地址改用 http://127.0.0.1:17890，不用 socks5h://。7897 是本次确认的端口；以后以软件实际配置为准。

## 在原服务器验证并登录

SOCKS 方式使用项目环境内的 PySocks。requirements-migration.txt 已补入固定版本，不向 Conda base 安装：

```bash
cd /YOUR_PROJECT_DIRECTORY
bash scripts/hf_with_proxy.sh socks5h://127.0.0.1:17890 --check-only
bash scripts/hf_with_proxy.sh socks5h://127.0.0.1:17890 hf auth login
bash scripts/hf_with_proxy.sh socks5h://127.0.0.1:17890 hf auth whoami
```

代理入口只对这一条命令及其子进程设置环境变量，保持官方 HF endpoint 和 TLS 校验；不修改系统 DNS、系统代理或其他训练进程。
命令会先做不带 HF token 的网络检查。匿名 HTTP 401 在这里说明连通了验证接口，不代表输入 token 已通过验证。登录成功后 whoami 应显示 PyCmg。

如果终端没有 PySocks，可以在项目环境执行：

```bash
.venv/bin/python -m pip install PySocks==1.7.1
```

若转发的是电脑 HTTP 代理，则把以上所有 socks5h://127.0.0.1:17890 换成 http://127.0.0.1:17890。
不要关闭证书校验，也不要把 token 发到聊天、粘到脚本或命令参数里。hf auth login 的交互提示会隐藏输入。
本项目使用 HF API 上传，登录时 Add token as git credential 可以选择 n。

## 验证后上传

```bash
bash scripts/hf_with_proxy.sh socks5h://127.0.0.1:17890 python scripts/manage_hub_release.py upload --namespace PyCmg
```

约 21 GiB 资产会经过电脑转发，占用其带宽。上传时保持电脑网络和 SSH 连接，不要休眠；转发关闭会中断上传，可建立相同转发后重试。
HF 成功不代表 GitHub 已授权。GitHub 的登录/推送需另行验证，不会复用 HF token。
