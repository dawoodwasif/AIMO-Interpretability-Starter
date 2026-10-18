"""Shared core used by BOTH the submission and the offline research pipeline.

Train/serve parity rule: any feature the classifier is trained on must be
computed by code in this package, so the bundle computes it identically.
"""
