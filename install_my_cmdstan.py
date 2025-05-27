# install_my_cmdstan.py
import cmdstanpy
try:
    cmdstanpy.install_cmdstan(overwrite=True, verbose=True)
    print("CmdStan installation successful!")
except Exception as e:
    print(f"CmdStan installation failed: {e}")
    print("Please ensure you have a C++ toolchain (g++, make) installed.")
    print("For manual installation, visit: https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html")

