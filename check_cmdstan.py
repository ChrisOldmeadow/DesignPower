# check_cmdstan.py
import cmdstanpy
import os

print(f"--- CmdStanPy Diagnostic ---")
print(f"CmdStanPy version: {cmdstanpy.__version__}")
print(f"Python executable: {os.sys.executable}") # Shows which Python is running this

try:
    cmdstan_path_found = cmdstanpy.cmdstan_path()
    print(f"CmdStan installation found by cmdstanpy at: {cmdstan_path_found}")
    
    # Further check: try to compile a very simple model
    model_code = "parameters {real y;} model {y ~ normal(0,1);}"
    stan_file_path = "temp_model.stan"
    with open(stan_file_path, "w") as f:
        f.write(model_code)
    
    print("Attempting to compile a dummy Stan model...")
    sm = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    print(f"Dummy model compiled successfully. Executable: {sm.exe_file}")
    os.remove(stan_file_path) # Clean up dummy .stan file
    if sm.exe_file and os.path.exists(sm.exe_file): # Clean up compiled model if it exists
        os.remove(sm.exe_file)
    if os.path.exists("temp_model.hpp"): # Clean up hpp file if it exists
         os.remove("temp_model.hpp")

except Exception as e:
    print(f"Error during diagnostic: {e}")
    print("CmdStanPy could not find or use a valid CmdStan installation.")
    print("Consider re-running `cmdstanpy.install_cmdstan(overwrite=True)` from this Python environment.")

print(f"--------------------------")

