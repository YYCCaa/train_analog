"""Preview render task run inside spawned pool workers. Deliberately does not
import FastAPI so worker start-up stays light."""
import base64
import io

from PIL import Image

import studio.backend  # noqa: F401  (puts repo root on sys.path)
import generate_synthetic_v7_mp as gen


def render_tile(cfg, seed):
    """Render one preview -> dict(seed int, png_b64 str of an "L" 256x256 PNG,
    tip_x/tip_y/angle_deg float). cfg: validated config dict; seed: int."""
    res = gen.render_image(cfg, seed)
    buf = io.BytesIO()
    Image.fromarray(res.image).save(buf, format="PNG")
    return {"seed": seed, "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "tip_x": float(res.tip_x), "tip_y": float(res.tip_y),
            "angle_deg": float(res.angle_deg)}
