"""Write predict_abdomenatlas_sek.py: the official R-Super MedFormer inference script with the 3D pad fix.

See run_medformer.sbatch. Run from third_party/R-Super/rsuper_train.
"""

s = open("predict_abdomenatlas.py").read()
z, x = "F.pad(tensor_img, (diff, diff, 0,0, 0,0))", "F.pad(tensor_img, (0,0, 0,0, diff, diff))"
head, tail = s.split("def pad_to_training_size", 1)
body, rest = tail.split("elif args.dimension == '2d'", 1)
assert body.count(z) == 1 and body.count(x) == 1, "upstream pad_to_training_size changed; re-check the fix"
body = body.replace(z, "@@Z@@").replace(x, z).replace("@@Z@@", x)
open("predict_abdomenatlas_sek.py", "w").write(head + "def pad_to_training_size" + body + "elif args.dimension == '2d'" + rest)
