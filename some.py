import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from place_horse import data
from scipy.stats import multivariate_normal
import math

epsilon = 0.05

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def normalise_vector(v):
    #v = stats.boxcox(v)[0]
    v = np.log(v)
    minim = np.min(v)
    maxim = np.max(v)
    v = (v-minim)/(maxim - minim)
    return v

if __name__ == '__main__':
    df = data.load()
    feat1 = normalise_vector(df['num_runners'])
    feat2 = normalise_vector(df['win_odds'])
    feat3 = normalise_vector(df['place_odds'])


    #### simple gaussian:
    p = feat1 * feat2 * feat3
    anomalies = df.rowslice(p < epsilon)
    plt.hist(p, bins=120); plt.show()

    #### multivariate_normal
    feats = np.vstack([feat1, feat2, feat3])
    covariance_matrix = np.matrix(np.corrcoef(feats))
    means = np.mean(feats,axis=1)
    var = multivariate_normal(mean=means, cov=covariance_matrix)

    feats = np.column_stack([feat1, feat2, feat3])
    var.pdf(feats)

    a = norm_pdf_multivariate(feats[0,:], means, covariance_matrix)

