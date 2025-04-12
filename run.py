import os
import subprocess
import platform


def find_python_command():
    for cmd in ['python', 'python3', 'py']:
        try:
            result = subprocess.run(
                [cmd, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None

python_cmd = find_python_command()
script = os.path.join("run", "test.py")
cmd = ""

if platform.system() == "Windows":
    activate_script = os.path.join("Scripts", "activate.bat")
    cmd = f'cmd /k "python3 {script}"'
else:
    activate_script = os.path.join("bin", "activate")
    cmd = f'/bin/bash -c "python3 {script}"'

subprocess.run(cmd, shell=True)
