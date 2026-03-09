def load_track(file_path):
    """
    Load track walls from a text file.

    Each line in the file:
    x1,y1,x2,y2
    """

    walls = []

    with open(file_path, "r") as f:
        for line in f.readlines():

            line = line.strip()

            if not line:
                continue

            x1, y1, x2, y2 = map(float, line.split(","))

            walls.append((x1, y1, x2, y2))

    return walls