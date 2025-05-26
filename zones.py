ZONES = {
    "general": [
        [(0, 0), (960, 0), (960, 1080), (0, 1080)]  # Left half
    ],
    "restricted": [
        [(960, 0), (1920, 0), (1920, 1080), (960, 1080)]  # Right half
    ]
}

ZONE_COLORS = {
    "general": (0, 255, 255),  # Yellow
    "restricted": (0, 0, 255)   # Red
}

ACCESS_LEVELS = {
    "niraj": "general",
    "sovit": "general"
}