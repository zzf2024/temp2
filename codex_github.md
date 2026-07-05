推荐在 GitHub Codespaces 终端里装 Codex CLI，因为 Codespace 是远端 Linux 环境，比深圳本地网络更容易稳定访问 OpenAI 服务。

步骤如下：

curl -fsSL https://chatgpt.com/codex/install.sh | sh

如果提示找不到 codex，补一下 PATH：

export PATH="$HOME/.local/bin:$PATH"

然后运行：

codex

首次运行会让你登录。Codespaces 属于远程/无头环境，优先用设备码登录：

codex login --device-auth

它会给你一个链接和一次性 code，你在本地浏览器打开链接、登录 ChatGPT、输入 code。OpenAI 官方文档也明确说 Codex CLI 可在终端运行，并支持 ChatGPT 账号或 API key 登录；远程/无头环境推荐 device code auth。(developers.openai.com) (developers.openai.com)

如果深圳本地打不开 chatgpt.com 登录页，有两个现实方案：

用合规的公司网络、代理或 VPN，让本地浏览器能打开 chatgpt.com 完成设备码登录。

改用 API key 登录或自动化方式，把密钥放进 GitHub Codespaces Secrets，不要写进仓库。API key 适合 CLI/CI 这类程序化工作流，但部分依赖 ChatGPT workspace/cloud 的功能可能不可用。(developers.openai.com)

常用验证：

codex --version

codex

如果你只是想在浏览器里用云端 Codex，也可以直接打开：

https://chatgpt.com/codex

但在深圳本地这仍然取决于你浏览器能否稳定访问 ChatGPT。Codespaces 里跑 CLI 通常是更实用的路线。



你这里没启动起来，最可能是因为 curl ... | sh 这种安装管道里直接启动 TUI，终端交互有时不会正常接管。不要纠结最后那个 Start Codex now? y，直接新开一个 Codespaces 终端手动启动。

在 /workspaces/costrict-enterprise 里执行：

which codex

codex --version

codex login --device-auth

codex

codex login --device-auth 会给你一个链接和一次性 code。你在本地浏览器打开链接登录，然后回到 Codespaces 再运行 codex。

如果 codex 找不到，执行：

export PATH="$HOME/.local/bin:$PATH"

codex --version

如果 codex login --device-auth 打不开或登录失败，深圳这边通常是本地浏览器访问 chatgpt.com 的网络问题，不是 Codespaces 安装问题。你可以换能访问 ChatGPT 的网络完成 device code 登录；或者运行 codex login，选择 API key 登录，但 API key 会按 OpenAI API 账号计费，且部分 ChatGPT workspace/cloud 功能不可用。

OpenAI 官方文档里的流程也是：安装后运行 codex，首次运行会要求登录；远程/无头环境推荐 codex login --device-auth。参考：

https://developers.openai.com/codex/cli

https://developers.openai.com/codex/auth

