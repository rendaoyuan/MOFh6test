import sys
import subprocess
import os
from datetime import datetime


def run_interactive_with_log(script_path, log_file=None):
    """
    交互式执行Python脚本并记录所有输入输出
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"session_{timestamp}.log"

    print(f"开始记录会话，日志文件: {log_file}")
    print(f"执行脚本: {script_path}")

    # 确保使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    # 打开日志文件
    with open(log_file, 'w', encoding='utf-8') as log:
        # 写入会话头信息
        log.write(f"{'=' * 70}\n")
        log.write(f"会话开始: {datetime.now()}\n")
        log.write(f"执行的脚本: {script_path}\n")
        log.write(f"Python路径: {sys.executable}\n")
        log.write(f"{'=' * 70}\n\n")

        # 启动Python进程（交互模式）
        proc = subprocess.Popen(
            [sys.executable, '-u', '-i', script_path],
            stdin=sys.stdin,  # 保持标准输入
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 将标准错误重定向到标准输出
            text=True,  # 文本模式
            encoding='utf-8',  # UTF-8编码
            bufsize=1,  # 行缓冲
            env=env  # 环境变量
        )

        print("\n" + "=" * 50)
        print("进入交互模式 (输入 Ctrl+C 退出)")
        print("=" * 50 + "\n")

        try:
            # 实时读取和显示输出
            while True:
                # 读取一行输出
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break

                if line:
                    # 显示到控制台
                    sys.stdout.write(line)
                    sys.stdout.flush()

                    # 记录到日志文件
                    log.write(line)
                    log.flush()

        except KeyboardInterrupt:
            print("\n\n检测到Ctrl+C，正在结束会话...")
            proc.terminate()

        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 等待进程结束
            proc.wait()

            # 写入会话结束信息
            log.write(f"\n{'=' * 70}\n")
            log.write(f"会话结束: {datetime.now()}\n")
            log.write(f"退出代码: {proc.returncode}\n")
            log.write(f"{'=' * 70}\n")

            print(f"\n会话已保存到: {log_file}")
            print(f"退出代码: {proc.returncode}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_interactive_with_log(sys.argv[1])
    else:
        print("用法: python run_and_log.py <python_script.py>")
        print("示例: python run_and_log.py main.py")