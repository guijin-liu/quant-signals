# -*- coding: utf-8 -*-
"""量化 → 腾讯云SCF 一键部署 v3（踩坑全固化，支持 1/2/3 号复用）
用法:
  python deploy_scf.py 2 init       # 打包+上传代码和层(2号)
  python deploy_scf.py 2 function   # 建函数(绑层)+等Active+建触发器
  python deploy_scf.py 2 invoke     # 手动触发验证
  python deploy_scf.py 2 all        # 一步到位
前提: 环境变量 TENCENTCLOUD_SECRETID / TENCENTCLOUD_SECRETKEY / PUSHPLUS_TOKEN

已固化的坑(2026-08-29 踩坑全集):
  1. COS桶名用 AppId(1440185993) 不是 Uin(100049534034)
  2. 层zip必须【根目录直放包】(numpy/pandas 直接在根), 不能加 python/ 前缀
  3. urllib3 必须 1.26.x (SCF Python3.9 是 OpenSSL 1.0.2, urllib3 v2 不兼容)
  4. 代码包用 base64 ZipFile 直传(小包<20MB), 不用 COS 引用(CodeSize=0 坑)
  5. 层发布走 COS 引用(大包>20MB 无法 base64), 需 Content.CosBucketRegion
  6. tccli cos 子命令 snake_case, scf 是 PascalCase; cos/CreateTrigger 不接受 --Region
  7. --cli-input-json 必须 file:// 路径, 且不能同带 --Region
  8. 环境变量 Key 禁用前缀 SCF_/QCLOUD_/TENCENTCLOUD_ (用 TX_ 代替)
  9. AutoCreateClsTopic=False 避开 CLS 日志服务未开通的权限报错
  10. 建触发器要等函数 Active(否则报"Creating 状态无法操作")
"""
import os, sys, io, zipfile, subprocess, base64, glob, shutil, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SECRET_ID = os.environ.get("TENCENTCLOUD_SECRETID", "")
SECRET_KEY = os.environ.get("TENCENTCLOUD_SECRETKEY", "")
PUSHPLUS = os.environ.get("PUSHPLUS_TOKEN", "")
if not SECRET_ID or not SECRET_KEY:
    print("错误: 请设置 TENCENTCLOUD_SECRETID / TENCENTCLOUD_SECRETKEY")
    sys.exit(1)

REGION = "ap-guangzhou"
APP_ID = "1440185993"
# 最小权限子账号(仅 COS 单桶读写)，函数内访问 COS 状态用，泄露也只影响一个桶
SUB_SECRET_ID = "AKIDEQceUAbqrdLuZzAC1bw2uYzDEG5g5HEx"
SUB_SECRET_KEY = "VnriLk9JSSM5arwmQflfaXXTEmFPROFu"
BUCKET = f"quant-signals-state-{APP_ID}"
LAYER_NAME = "quant-deps"
TCCLI = r"C:\Program Files\Python314\Scripts\tccli.exe"
CRON = "0 */5 9-15 * * MON-FRI *"   # 周一到五 9:00-15:00 每5分钟(覆盖交易时段)

# ── 各量化程序配置 ──
PROGRAMS = {
    "2": {
        "func": "quant-scan-2",
        "handler": "scf_cloud2.main_handler",
        "modules": ["cloud2_function.py", "em_client.py", "mx_fetcher.py", "fixed_pool_2.py",
                    "backtest_miner2.py", "fund_eval.py", "hot_pool.py", "scf_cloud2.py"],
        "data": ["rules2.json", "data/per_stock_rules.json"],
        "code_zip": "scf_cloud2_code.zip",
    },
    "1": {
        "func": "quant-scan-1",
        "handler": "scf_cloud1.main_handler",
        "modules": ["cloud_function.py", "em_client.py", "mx_fetcher.py", "stock_pool.py",
                    "backtest_miner.py", "fund_eval.py", "hot_pool.py", "scf_cloud1.py"],
        "data": [],
        "code_zip": "scf_cloud1_code.zip",
    },
    "3": {
        "func": "quant-scan-3",
        "handler": "scf_cloud3.main_handler",
        "modules": ["cloud3_function.py", "em_client.py", "mx_fetcher.py", "hot_pool.py",
                    "dynamic_pool.py", "backtest_miner2.py", "fund_eval.py", "scf_cloud3.py"],
        "data": ["rules2.json", "data/per_stock_rules.json"],
        "code_zip": "scf_cloud3_code.zip",
    },
}

# manylinux 依赖（SCF Python3.9 = Linux x86_64；scipy 仅回测用，实盘不绑）
DEPS = ["pandas", "numpy", "requests", "cos-python-sdk-v5"]


def run(cmd, ok="OK", check=True):
    print(f"$ {' '.join(cmd)[:110]}")
    r = subprocess.run(cmd)   # 列表参数+继承输出，避开沙箱管道EPERM和shell转义坑
    if r.returncode != 0:
        print(f"  ✗ 退出码 {r.returncode}")
        if check:
            sys.exit(f"失败: {' '.join(cmd)[:80]}")
        return r
    print(f"  ✓ {ok}")
    return r


def tc(args, ok="OK", check=True, region=True):
    cmd = [TCCLI] + (args if isinstance(args, list) else args.split())
    if region and "--cli-input-json" not in cmd:
        cmd += ["--Region", REGION]
    return run(cmd, ok, check)


def run_out(cmd, outfile):
    print(f"$ {' '.join(cmd)[:100]}")
    with open(outfile, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


def pack_code(p):
    """纯业务代码打包(依赖走层)，含运行期数据文件"""
    code_zip = p["code_zip"]
    if os.path.exists(code_zip):
        os.remove(code_zip)
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for m in p["modules"]:
            if os.path.exists(m):
                z.write(m, m)
        for d in p["data"]:
            if os.path.exists(d):
                z.write(d, d)
    print(f"代码包 {code_zip}: {os.path.getsize(code_zip)/1024:.2f} KB")


def build_layer():
    """依赖层：根目录直放包(不加python前缀) + urllib3 固定1.26"""
    if os.path.exists("scf_layer.zip"):
        os.remove("scf_layer.zip")
    dep_dir = "_deps_linux"
    if not glob.glob(os.path.join(dep_dir, "*.whl")):
        os.makedirs(dep_dir, exist_ok=True)
        cmd = ("pip download --platform manylinux2014_x86_64 --only-binary=:all: "
               "--python-version 39 --implementation cp --abi cp39 "
               "-i https://pypi.tuna.tsinghua.edu.cn/simple "
               "-d " + dep_dir + " " + " ".join(DEPS) + " urllib3==1.26.18")
        run(cmd, "下载manylinux依赖")
    else:
        print(f"依赖已缓存 ({len(glob.glob(dep_dir + '/*.whl'))} 个wheel)")
    layer_root = "_layer_root"
    if os.path.exists(layer_root):
        shutil.rmtree(layer_root)
    os.makedirs(layer_root)
    for whl in glob.glob(os.path.join(dep_dir, "*.whl")):
        with zipfile.ZipFile(whl) as z:
            z.extractall(layer_root)
    with zipfile.ZipFile("scf_layer.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(layer_root):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, layer_root))
    print(f"层包 scf_layer.zip: {os.path.getsize('scf_layer.zip')/1024/1024:.1f} MB")


def upload_cos(local, key):
    tc(["cos", "upload", "--bucket", BUCKET, "--local_path", os.path.abspath(local),
        "--cos_key", key], f"COS上传 {key}", region=False)


def publish_layer():
    """发布层(COS引用，大包>20MB无法base64)"""
    payload = json.dumps({
        "LayerName": LAYER_NAME,
        "CompatibleRuntimes": ["Python3.9"],
        "Content": {"CosBucketName": BUCKET, "CosObjectName": "layer/scf_layer.zip",
                    "CosBucketRegion": REGION},
    }, ensure_ascii=False)
    jf = "_layer_payload.json"
    with open(jf, "w", encoding="utf-8") as f:
        f.write(payload)
    out = "_layer_pub.json"
    run_out([TCCLI, "scf", "PublishLayerVersion", "--cli-input-json",
             "file://" + os.path.abspath(jf)], out)
    with open(out, encoding="utf-8") as f:
        v = json.loads(f.read()).get("LayerVersion", 0)
    # 等层 Active
    for _ in range(12):
        time.sleep(3)
        o2 = "_layer_stat.json"
        run_out([TCCLI, "scf", "GetLayerVersion", "--LayerName", LAYER_NAME,
                 "--LayerVersion", str(v)], o2)
        try:
            if json.loads(open(o2, encoding="utf-8").read()).get("Status") == "Active":
                print(f"  ✓ 层 v{v} Active")
                return v
        except Exception:
            pass
    print(f"  ⚠️ 层 v{v} 未确认 Active，仍使用")
    return v


def wait_active(p):
    for _ in range(20):
        time.sleep(5)
        o = "_fn_stat.json"
        run_out([TCCLI, "scf", "GetFunction", "--FunctionName", p["func"],
                 "--Namespace", "default"], o)
        try:
            if json.loads(open(o, encoding="utf-8").read()).get("Status") == "Active":
                print("  ✓ 函数 Active")
                return True
        except Exception:
            pass
    return False


def create_function(p, layer_version):
    env_vars = [
        {"Key": "PUSHPLUS_TOKEN", "Value": PUSHPLUS},
        {"Key": "STATE_BUCKET", "Value": BUCKET},
        {"Key": "COS_REGION", "Value": REGION},
        {"Key": "TX_SECRETID", "Value": SUB_SECRET_ID},
        {"Key": "TX_SECRETKEY", "Value": SUB_SECRET_KEY},
    ]
    code_b64 = base64.b64encode(open(p["code_zip"], "rb").read()).decode()
    payload = json.dumps({
        "FunctionName": p["func"],
        "Namespace": "default",
        "Handler": p["handler"],
        "Runtime": "Python3.9",
        "Timeout": 300,
        "MemorySize": 256,
        "Role": "SCF_QcsRole",
        "AutoCreateClsTopic": "False",
        "Code": {"ZipFile": code_b64},
        "Environment": {"Variables": env_vars},
        "Layers": [{"LayerName": LAYER_NAME, "LayerVersion": layer_version}],
    }, ensure_ascii=False)
    jf = "_func_payload.json"
    with open(jf, "w", encoding="utf-8") as f:
        f.write(payload)
    tc(["scf", "CreateFunction", "--cli-input-json", "file://" + os.path.abspath(jf)],
       "函数创建")
    wait_active(p)
    tc(["scf", "CreateTrigger", "--FunctionName", p["func"], "--Namespace", "default",
        "--TriggerName", "every5min", "--Type", "timer", "--TriggerDesc", CRON, "--Enable", "1"],
       "定时触发器", region=False)


def invoke(p):
    tc(["scf", "Invoke", "--FunctionName", p["func"], "--Namespace", "default",
        "--InvocationType", "Sync", "--ClientContext", '{"test":1}'], "手动触发")


if __name__ == "__main__":
    prog = sys.argv[1] if len(sys.argv) > 1 else "2"
    action = sys.argv[2] if len(sys.argv) > 2 else "all"
    if prog not in PROGRAMS:
        print(f"未知程序 {prog}，可选: {list(PROGRAMS)}")
        sys.exit(1)
    p = PROGRAMS[prog]
    if action == "init":
        tc(["cos", "create_bucket", "--bucket", BUCKET], "COS桶已建", check=False, region=False)
        pack_code(p)
        build_layer()
        upload_cos(p["code_zip"], f"code/{p['code_zip']}")
        upload_cos("scf_layer.zip", "layer/scf_layer.zip")
    elif action == "layer":
        publish_layer()
    elif action == "function":
        v = publish_layer()
        create_function(p, v)
    elif action == "invoke":
        invoke(p)
    elif action == "all":
        tc(["cos", "create_bucket", "--bucket", BUCKET], "COS桶已建", check=False, region=False)
        pack_code(p)
        build_layer()
        upload_cos(p["code_zip"], f"code/{p['code_zip']}")
        upload_cos("scf_layer.zip", "layer/scf_layer.zip")
        v = publish_layer()
        create_function(p, v)
        invoke(p)
    else:
        print(__doc__)
