
vectors = [{0: 1, 1: 1}, {0: 3, 2: 2}]
cluster_ids = [0, 0]

centroids = []
unique_clusters = set(cluster_ids)

for cluster_id in unique_clusters:
    cluster_vectors = []
    for i in range(len(vectors)):
        if cluster_ids[i] == cluster_id:
            cluster_vectors.append(vectors[i])

    common_keys = set()  # TODO: understand this bit
    for v in cluster_vectors:
        common_keys.update(v.keys())

    centroid = {}
    for key in common_keys:
        values = []
        for v in cluster_vectors:
            values.append(v.get(key, 0))
        avg_value = sum(values) / len(cluster_vectors)
        centroid[key] = avg_value

    centroids.append(centroid)

print(centroids)
