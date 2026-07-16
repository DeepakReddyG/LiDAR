

if abs(z - surface_z[row, col]) < 3.0:
    # ON surface → trust SAM3
    label = label_grid[row, col]
else:
    # BELOW surface → LiDAR features
    if hag > 6 and ndvi > 0.82:
        label = TREE      # forest green
    elif hag < 1 and ndvi < 0.15:
        label = PAVEMENT  # gray
    else:
        label = GRASS     # lawn green