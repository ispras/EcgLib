from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
import sys

class InstallSequential(install):
    """Custom install command that installs the cauchy extension after setup."""
    
    def run(self):
        install.run(self)
        # Now that everything is installed, install cauchy
        cauchy_path = os.path.join(os.path.dirname(__file__), "src/ecglib/models/architectures/sssd/extensions/cauchy")
        print("Installing cauchy extension...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", cauchy_path])

setup(
    cmdclass={"install": InstallSequential},
)