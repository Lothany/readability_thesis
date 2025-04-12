import os
import subprocess
import platform

script = os.path.join("run", "test.py")
cmd = ""

if platform.system() == "Windows":
    activate_script = os.path.join("Scripts", "activate.bat")
    cmd = f'cmd /k "python {script}"'
else:
    activate_script = os.path.join("bin", "activate")
    cmd = f'/bin/bash -c "python {script}"'

subprocess.run(cmd, shell=True)
