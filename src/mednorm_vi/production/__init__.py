"""The MedNorm-VI production system: one architecture, one inference path.

`pipeline.run` is the whole system. This package intentionally re-exports nothing heavy: the
retired inference package became unusable because its init imported the entire stack, and a
package init is the wrong place to decide what loads.
"""

__all__ = ["pipeline"]
