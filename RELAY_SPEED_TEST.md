# UCloud → AutoDL 中转链路测速

测速工具是项目根目录中的 `relay_speed_test.py`，只依赖 Python 3 标准库。

> 测速阶段使用 HTTP 和随机测试文件，不要上传真实音轨。正式中转必须配置 HTTPS。

## 1. 生成临时令牌

在本机 PowerShell 执行：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

复制输出的令牌。测试期间，UCloud、本机和 AutoDL 必须使用同一个令牌。不要把令牌提交到 Git。

## 2. 在 UCloud 启动测速服务

先把 `relay_speed_test.py` 上传到 UCloud，例如：

```powershell
scp .\relay_speed_test.py root@<UCLOUD公网IP>:/root/
```

登录 UCloud 后执行：

```bash
export RELAY_TEST_TOKEN='<刚才生成的令牌>'
python3 /root/relay_speed_test.py serve \
  --host 0.0.0.0 \
  --port 8766 \
  --storage /root/relay-speed-test
```

在 UCloud 防火墙/安全组中临时放行 TCP `8766`。测速完成后应关闭此端口。

可在本机浏览器访问以下地址检查服务：

```text
http://<UCLOUD公网IP>:8766/health
```

正常响应：

```json
{"ok": true, "service": "subtitle-relay-speed-test"}
```

## 3. 测试本机 → UCloud 上传速度

在项目目录打开 PowerShell：

```powershell
$env:RELAY_TEST_TOKEN='<刚才生成的令牌>'
python .\relay_speed_test.py generate --file .\relay-test.bin --size-mb 100
python .\relay_speed_test.py upload `
  --url http://<UCLOUD公网IP>:8766 `
  --file .\relay-test.bin
```

记录最后显示的 `MiB/s` 和 `Mbps`。

## 4. 测试 UCloud → AutoDL 下载速度

把同一个测速脚本传到 AutoDL：

```powershell
scp -P <AUTODL端口> .\relay_speed_test.py root@<AUTODL主机>:/root/
```

在 AutoDL 终端执行：

```bash
export RELAY_TEST_TOKEN='<刚才生成的令牌>'
python3 /root/relay_speed_test.py download \
  --url http://<UCLOUD公网IP>:8766 \
  --name relay-test.bin \
  --output /root/relay-test-downloaded.bin
```

下载完成后脚本会自动核对 SHA-256。没有出现校验错误即表示文件完整。

## 5. 清理测试文件

在本机执行：

```powershell
python .\relay_speed_test.py delete `
  --url http://<UCLOUD公网IP>:8766 `
  --name relay-test.bin
Remove-Item -LiteralPath .\relay-test.bin
```

在 AutoDL 删除下载副本：

```bash
rm -f /root/relay-test-downloaded.bin
```

回到 UCloud 按 `Ctrl+C` 停止测速服务，并关闭安全组的 TCP `8766` 端口。

## 结果判断

- UCloud → AutoDL 低于 `0.5 MiB/s`：中转价值不大。
- `1～3 MiB/s`：适合在 GPU 开机前预上传，可明显减少付费等待。
- `3 MiB/s` 以上：适合集成到字幕工作室。
- 本机 → UCloud 即使较慢，只要在 AutoDL 关机时预上传，也不会浪费 GPU 费用。
