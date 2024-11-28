import os
import time
import shutil
import subprocess

watch_path = "./results"
max_dirs = 1
check_interval = 60
repo_id = "t5-xlarge-ko-kb"


def get_upload_command(repo_id, local_path, revision):
    return f"huggingface-cli upload {repo_id} {local_path} --revision {revision}"

def get_oldest_directory(path):
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return None
    oldest_dir = min(subdirs, key=os.path.getctime)
    return oldest_dir


def main():
    while True:
        subdirs = [d for d in os.listdir(watch_path) if os.path.isdir(os.path.join(watch_path, d))]
        if len(subdirs) > max_dirs:
            oldest_dir = get_oldest_directory(watch_path)
            if oldest_dir:
                revision = os.path.basename(oldest_dir)
                upload_command = get_upload_command(repo_id, oldest_dir, revision)
                try:
                    subprocess.run(upload_command, shell=True, check=True)
                    shutil.rmtree(oldest_dir)
                    print(f"Uploaded and removed: {oldest_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to execute upload command: {upload_command}")
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

        time.sleep(check_interval)


if __name__ == "__main__":
    main()