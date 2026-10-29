"""Original five-stage geometry fitting schedule for tissue demonstrations."""

TISSUE_SCHEDULE = [
    ('step1_rigid', 500, .01, False),
    ('step2_rigid', 500, .005, False),
    ('step3_non-rigid', 500, .01, True),
    ('step4_non-rigid', 1500, .005, True),
    ('step5_non-rigid', 2000, .0001, True),
]
