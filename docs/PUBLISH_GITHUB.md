# Publish To GitHub (Sciencewisdom)

This repo is prepared for GitHub. Large artifacts are excluded via `.gitignore`.

## Option A: SSH (recommended)

1) Ensure you have an SSH key and it's added to GitHub:

- Check:
  ```bash
  ls -lah ~/.ssh
  ```
- If needed:
  ```bash
  ssh-keygen -t ed25519 -C "sciencewisdom@users.noreply.github.com"
  ```
- Add the public key (`~/.ssh/id_ed25519.pub`) to GitHub Settings -> SSH keys.

2) Create a new GitHub repo under `Sciencewisdom`, e.g. `imu-sota-solution`.

3) Push:

```bash
cd /tmp/项目/imu_sota_solution
git remote add origin git@github.com:Sciencewisdom/imu-sota-solution.git
git branch -M main
git push -u origin main
```

## Option B: HTTPS + PAT

1) Create a Personal Access Token (classic) with `repo` scope.

2) Push:

```bash
cd /tmp/项目/imu_sota_solution
git remote add origin https://github.com/Sciencewisdom/imu-sota-solution.git
git branch -M main
git push -u origin main
```

When prompted for password, paste the PAT.

## Notes

- Do not commit model weights or `dist_submit/` artifacts to GitHub; keep them as release assets or external storage.
- If you need a reproducibility bundle, attach `report_materials.tgz` and `submission_package_lite.tgz` as release files.
