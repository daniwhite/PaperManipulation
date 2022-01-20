name = "partial_sphere.obj"

header_lines = 4

ordered_prefixes = ["v", "vn", "f"]
skipped_prefixes = {"o"}
data = {
    "v": [],
    "vn": [],
    "f": [],
}

outlines = []
with open("models/" + name, 'r') as f:
    for i, l in enumerate(f.readlines()):
        if i < header_lines:
            outlines.append(l)
            continue
        prefix = l.split(" ")[0]
        if prefix in skipped_prefixes:
            continue
        data[prefix].append(l)

for pref in ordered_prefixes:
    for l in data[pref]:
        outlines.append(l)

with open("models/" + name, 'w') as f:
    f.writelines(outlines)
