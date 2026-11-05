import os, sys, time, types
handler_dir, model_home, pictures = sys.argv[1], sys.argv[2], sys.argv[3:]
os.environ["U2NET_HOME"] = model_home
mark = time.time(); import rembg; print(f"import rembg (first, cold cache): {time.time() - mark:.1f} s")
sys.modules["runpod"] = types.ModuleType("runpod")
sys.path.insert(0, handler_dir)
import numpy as np
from PIL import Image
import handler
session = rembg.new_session("u2net")
for path in pictures:
    img = Image.open(path).convert("RGBA")
    if img.getextrema()[3][0] < 255:
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB").convert("RGBA")          # plain opaque picture, as builds send
    mark = time.time(); ours = np.array(handler._cutout(img)); t_ours = time.time() - mark
    mark = time.time(); theirs = np.array(rembg.remove(img.convert("RGBA"), session=session)); t_theirs = time.time() - mark
    diff = np.abs(ours.astype(int) - theirs.astype(int))
    print(f"{os.path.basename(path)} {img.size}: identical={np.array_equal(ours, theirs)} max_diff={diff.max()} "
          f"alpha_mean={ours[:, :, 3].mean():.1f} | ours {t_ours:.2f} s, rembg {t_theirs:.2f} s")
