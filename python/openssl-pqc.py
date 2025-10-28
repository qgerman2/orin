import subprocess

while True:
    subprocess.run(["openssl", "speed", "-multi", "8", "-seconds", "1", "ML-DSA-65"],
                   env={"LD_LIBRARY_PATH": "/"})

    subprocess.run(["openssl", "speed", "-multi", "8", "-seconds", "1", "ML-KEM-768"],
                   env={"LD_LIBRARY_PATH": "/"})
