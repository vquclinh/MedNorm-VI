"""Pretrained hybrid semantic linking (Audit 0071).

Model-free by design: representation, the null gate and hybrid fusion are all deterministic
and testable on a laptop, while the encoders and rerankers they feed run on Colab GPUs.
"""
