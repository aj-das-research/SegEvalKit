"""Interactive viewer: CT, reference and model predictions in the browser (``segevalkit view``).

    from segevalkit.app import ViewerSession, serve, export_html
    s = ViewerSession("ct.nii.gz", ref="labels/", preds={"nnU-Net": "pred_nnunet.nii.gz"},
                      labels={"liver": 1, "tumour": 2})
    serve(s)                               # http://localhost:8765
    export_html(s, "case.html")            # one self-contained file

Built on NiiVue (BSD-2-Clause, vendored in ``static/``): multiplanar and
montage views, a rotatable 3D view with surface meshes, click-to-highlight
structures and lesions, error colouring against the reference (TP violet,
FN orange, FP teal), and a synchronised side-by-side comparison of models.
"""

from .demo import demo_session, make_phantom
from .export import export_html
from .server import make_server, serve
from .session import ERROR_CODES, ViewerSession

NIIVUE_VERSION = "0.69.0"

__all__ = ["ViewerSession", "serve", "make_server", "export_html", "demo_session", "make_phantom",
           "ERROR_CODES", "NIIVUE_VERSION"]
