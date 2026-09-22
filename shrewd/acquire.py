"""Pick which pool rows are worth a teacher call. The ones the current tfidf probe is least
sure about, spread across clusters. Measured at 1.1-2.3x fewer labels than file order for
the same student accuracy. The probe's picks were about as useful to the embedding
student as random order, so the probe does not commit you to shipping tfidf.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from shrewd.students import TfidfStudent


def default_batch(n_pool):
    return int(min(500, max(50, n_pool // 10)))


def fit_probe(train_texts, train_labels, seed):
    probe = TfidfStudent(seed=seed)
    probe.fit(train_texts, train_labels)
    return probe


def select(probe, candidates, batch, seed):
    """Return indices into `candidates` for the next batch.

    Rank by margin between the top two probabilities (small margin = unsure), take the
    4x most uncertain as a shortlist, cluster the shortlist and keep the most uncertain
    row from each cluster.
    """
    if batch >= len(candidates):
        return list(range(len(candidates)))
    proba = probe.predict_proba(candidates)
    top2 = np.sort(proba, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    order = np.argsort(margin)
    shortlist = order[: min(len(candidates), 4 * batch)]
    if len(shortlist) <= batch:
        return [int(i) for i in shortlist]
    features = probe._pipe.named_steps["features"].transform([candidates[i] for i in shortlist])
    dims = min(64, features.shape[1] - 1, len(shortlist) - 1)
    if dims < 2:
        return [int(i) for i in order[:batch]]
    z = TruncatedSVD(dims, random_state=seed).fit_transform(features)
    clusters = KMeans(n_clusters=batch, n_init=3, random_state=seed).fit_predict(z)
    picked = []
    for c in range(batch):
        members = np.where(clusters == c)[0]
        if len(members):
            picked.append(int(shortlist[members[np.argmin(margin[shortlist[members]])]]))
    if len(picked) < batch:  # empty clusters: top up by uncertainty
        taken = set(picked)
        picked += [int(i) for i in order if int(i) not in taken][: batch - len(picked)]
    return picked
