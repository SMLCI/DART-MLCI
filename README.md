# DART-MLCI — Media Assets

This branch hosts media files (thumbnails, GIFs) referenced by the main
README via `raw.githubusercontent.com` URLs. Kept on a separate branch so
they don't bloat the PyPI wheel/sdist.

Source of truth lives on `main`. Update via:

```bash
git checkout media
# replace docs/assets/...
git add docs/assets
git commit -m "media: ..."
git push origin media
```
