"""The coarse question vocabulary — one source of truth (docs/POLICY.md §2).

inspect / run / modify is the proven 3-class shape; fine tool identity is
label noise (docs/DESIGN.md, 'Coarse questions'). The proxy's observe leg, the seed
fitter and the observe replay must all map tools through the SAME table.
"""
CLASSES = ["inspect", "run", "modify"]
MERGE = {"read": "inspect", "grep": "inspect", "glob": "inspect",
         "bash": "run", "edit": "modify", "write": "modify"}
