# The `docs-assets` release: where the images of the documentation are

The images and animations of the README, UserGuide, AnalysisGallery and `docs/` are **not in the repository**. They are files attached to one GitHub release of
`FlexTRKR/PyFLEXTRKR`, the pre-release [`docs-assets`](https://github.com/FlexTRKR/PyFLEXTRKR/releases/tag/docs-assets) (43 files, 55 MB in October 2026), and the markdown links to them:

```text
https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/<file name>
```

This page says what that means, and how to add, replace and remove a file. The commands were tested on a temporary release in the development fork (the results are in the last section).

## 1. How a GitHub release holds files

- A **release** is a page of a repository with a title, notes, a git tag and **assets**: files that you attach to it. An asset is not part of the git history: it is stored by GitHub next to the repository, so it is not in a
  clone, does not make the repository larger, and is not copied to forks (a fork starts with no releases; the development fork has none).
- The **URL of an asset** is `https://github.com/<owner>/<repo>/releases/download/<tag>/<file name>`. It stays the same as long as the tag and the file name stay the same, whatever happens to the file behind it. A README, a pull request
  and an old version of the README that link it all show **whatever the file is now**.
- **Each asset is a separate file.** Adding, replacing or deleting one asset does not touch the others, and it does not make a copy of the release. GitHub keeps **no versions**: a replaced file is gone
  (the old picture exists only where you saved it, or can be rendered again).
- **Limits** (GitHub documentation): each file under 2 GiB, up to 1000 files per release, no limit on the total size of a release or on the bandwidth of the downloads.
- **Who can change it**: people with write access to `FlexTRKR/PyFLEXTRKR` (the `gh` commands below). A contributor who works from a fork sends the file to a maintainer (section 4.6).
- **Immutable releases**: a release for which the repository setting "immutable releases" is on cannot get, lose or change assets after it is published. `docs-assets` is not immutable (the commands below work); do not switch that setting on for it.
- The release is marked **pre-release**, so it does not take the label "Latest" from the real releases (`v2025.05.0`, ...), and its tag `docs-assets` (a plain tag on the commit that was `main` when it was made) is not a version.

**Does every update make a new copy of the whole release? No.** One replaced file is one deleted and one uploaded file, 1-2 seconds for a small image; the other 42 files keep their id and their date (tested).

**Where is the master copy?** Only on GitHub, in that release. The files are not in git, and the folder on the machine where they were made may be temporary (scratch). To get all of them back:
`gh release download docs-assets --repo FlexTRKR/PyFLEXTRKR --dir docs_assets_backup` (55 MB). A figure can also be made again with the script that made it (`docs/visualization/`).

## 2. What is in it

43 files with unique names, all in one flat list: the images of the README, UserGuide and AnalysisGallery that used to be on the NERSC portal (cover image, workflow GIFs, the TGW_MCS and saag galleries), the figures and animations of
`docs/visualization/` and `visualization/README.md`, and the animations of the README (`mcs_global_imerg_2days.gif`, `mcs_sahel_2days.gif`, ...). Which markdown file uses which image:
`grep -rn "docs-assets/<file name>" --include=*.md .`

## 3. The commands

Always give the repository: in a clone of the development fork a bare `gh release ...` looks at the fork, which has no such release ("release not found").

| what | command |
|---|---|
| add a file | `gh release upload docs-assets NEW_FILE --repo FlexTRKR/PyFLEXTRKR` |
| replace a file of the same name | `gh release upload docs-assets FILE --clobber --repo FlexTRKR/PyFLEXTRKR` |
| remove a file | `gh release delete-asset docs-assets FILE_NAME --yes --repo FlexTRKR/PyFLEXTRKR` |
| download all files, or some | `gh release download docs-assets --repo FlexTRKR/PyFLEXTRKR --dir DIR`, with `--pattern 'mcs_*.gif'` for some |

List the files (name, bytes, date), and check that the URL serves the file that you have:

```bash
gh release view docs-assets --repo FlexTRKR/PyFLEXTRKR --json assets --jq '.assets[] | [.name, .size, .updatedAt[0:10]] | @tsv'
curl -sL https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/FILE_NAME | sha256sum
sha256sum FILE_NAME                      # the same number
```

## 4. Step by step

### 4.1 Add a new image

1. Make the file with a **new, unique name** of letters, digits, `_`, `-` and `.` (no spaces or brackets: GitHub rewrites them, section 5). PNG files 24-bit, GIF files as small as the README needs
   (settings and sizes: [animations.md](visualization/animations.md)).
2. Upload it: `gh release upload docs-assets my_new_figure.png --repo FlexTRKR/PyFLEXTRKR`. If the name exists, nothing is changed and the command says `asset under the same name already exists`.
3. Check it: `curl -sIL https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/my_new_figure.png | grep -i -E '^HTTP|content-length'` ends with `HTTP/2 200` and the size of your file.
4. Link it in the markdown: `![what it shows](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/my_new_figure.png)` (a GIF the same way).
5. Run `pytest tests/test_docs.py`: it accepts only image URLs of this release (or files in the repository), and it checks the relative links and the table rows. It does not use the network, so step 3 is the check that the file exists.

### 4.2 Replace an existing image (the same name)

Use this to correct a picture that every page should show corrected, e.g. a typo in a figure.

1. Keep the new file with **exactly the old name**, and keep a copy of the old one if you may need it back (`gh release download docs-assets --pattern NAME --repo FlexTRKR/PyFLEXTRKR`): **`--clobber` deletes the old file first, and if the upload
   fails the old one is lost** (gh says so in its help).
2. `gh release upload docs-assets NAME --clobber --repo FlexTRKR/PyFLEXTRKR`.
3. **Everything that links it shows the new picture**: the README and docs of `main`, of every branch, of every old tag, and the pull requests that were merged long ago. If an old page should keep its picture, do not replace: section 4.3.
4. **The old picture can still be served for about two minutes** after the upload (in the test the URL answered the old bytes for 102 s and the new bytes from 129 s on; the new file is there at once under its id). Wait, then compare
   `curl -sL URL | sha256sum` with `sha256sum NAME`, and reload the page without the cache (Ctrl+Shift+R) before you decide that it did not work.
5. Nothing in the markdown changes, so there is nothing to commit.

### 4.3 A new version that the old pages should not get

Upload it under a **new name** (`projections_v2.png`, as in 4.1) and change the link where the new picture should appear. The old file stays for the old pages. A change that is part of a pull request should be done like this:
the reviewers see the new picture in the pull request, `main` keeps its picture until the merge, and the old file is deleted afterwards (4.5).

### 4.4 Rename a file

There is no rename. Upload the file under the new name (4.1), change the links (`grep -rn "docs-assets/OLD" --include=*.md .`), and delete the old file (4.5) when no page that you care about uses it any more: its URL then answers 404 at once.

### 4.5 Remove a file

`gh release delete-asset docs-assets OLD_NAME --yes --repo FlexTRKR/PyFLEXTRKR`; first `grep -rn "docs-assets/OLD_NAME" --include=*.md .` shows the pages that link it.

### 4.6 You have no write access to the repository

A contributor shows the picture in the pull request discussion (dragged into a comment, GitHub hosts it there) or sends it to a maintainer, who uploads it to the release (4.1) and tells the final URL. The markdown in the
pull request has to use the release URL, because `tests/test_docs.py` accepts only that one (and the pictures of a comment are not in the repository or under its control).

### 4.7 Try a figure before it is public

Make a release in your own fork with a tag of your own (`gh release create docs-assets-try --prerelease --title try --notes try fig.png --repo YOUR_USER/YOUR_FORK`), link
`https://github.com/YOUR_USER/YOUR_FORK/releases/download/docs-assets-try/fig.png` while you work (`tests/test_docs.py` rejects it, so it is only for looking), upload to the real release (4.1) and change the link before the pull request, and remove your try release
(`gh release delete docs-assets-try --cleanup-tag --yes --repo YOUR_USER/YOUR_FORK`).

### 4.8 The release is lost or damaged

Take the files from a backup (`gh release download`, section 1) or render the figures again, and make the release again with the same tag and title, the same flags and the same file names
(`gh release create docs-assets --prerelease --latest=false --title "Documentation images and animations" --repo FlexTRKR/PyFLEXTRKR FILES...`): the URLs are then the old ones.

## 5. Names and formats

- **Names**: unique in the release; only letters, digits, `_`, `-` and `.`. GitHub changes the others: `my plot (1).png` becomes `my.plot.1.png` (tested), and the URL with the original name answers 404.
- **Formats**: PNG for figures (24-bit, the true shape of the map), GIF for animations (the docs: 16 frames, 480-720 pixels wide, 1.6-2.3 MB; the README: 800-1000 pixels, 6-12 MB; 256 colors, no dither). How the files are made:
  [animations.md](visualization/animations.md), [visualization/README.md](../visualization/README.md).
- The files are served as `application/octet-stream` with a download header; a browser shows them in an `<img>` tag and a GIF plays, as the README of the repository does.

## 6. What not to do

- Do not delete the release or its tag, move the tag, or make the release again with another tag: every link in every old README breaks. Do not mark it "Latest" and do not give it a version-like tag (a package or conda-forge update bot could take it for a version).
- Do not use `--clobber` without the new file in hand and the old one saved.
- Do not turn on "immutable releases" for it (4.2 and 4.1 stop working).
- Do not commit images to the repository (they stay in every clone for ever, in the git history, even if you delete them later); `tests/test_docs.py` will not find them, but the review should.

## 7. How this was tested

The commands of sections 3-5 were run on 2026-10-20 on a temporary pre-release in the development fork (`WACCEM/PyFLEXTRKR-dev`, deleted afterwards), with 16 x 16 pixel PNG files. The real `docs-assets` release was not touched.

| test | result |
|---|---|
| upload a file that exists, without `--clobber` | refused: `asset under the same name already exists: [a.png]`, nothing changed |
| `--clobber` of `a.png` with other bytes | 1.8 s; the file has a new id and a new date; `b.png` and `c.png` keep their ids and dates; the number of files is the same |
| the URL right after the replacement | the old bytes right after the upload in both runs; in the run that polled every 5 s the new bytes came after 102-129 s; the new file is served at once by its id (`gh api -H "Accept: application/octet-stream" repos/OWNER/REPO/releases/assets/ID`) |
| a name with spaces and brackets | stored as `my.plot.1.png` |
| `gh release delete-asset` | the URL answers 404 at once |
| `gh release download` (all, and `--pattern`) | works: all files, or only those that match |
| the response of the URL | a 302 with `cache-control: no-cache` to a signed address, then `200` with `content-type: application/octet-stream` and `content-disposition: attachment` |
| the fork | has no releases of its own; the releases of `FlexTRKR/PyFLEXTRKR` are not copied |
