import subprocess

def update_networkx():
    try:
        subprocess.check_call(['pip', 'install', '--upgrade', 'networkx'])
        print("Successfully updated networkx to the latest version.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while updating networkx:", e)

