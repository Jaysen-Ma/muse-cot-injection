"""Shared, dependency-free helpers for the cot-injection experiments on Muse Glimmer 30B.

Only small pieces of logic that several scripts share live here (repo paths,
the ATEM reasoning-channel prefill, the StrongREJECT-style judge parser and the
category-holdout split). Everything here is stdlib-only so it can be unit
tested without a GPU or a model server.
"""

__all__ = ["paths", "atem", "judge", "splits"]
