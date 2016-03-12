import numpy as np
from scipy import linalg
from numpy import dot

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
	"""
	Decompose X to A*Y
	"""
	eps = 1e-5
	print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
	#X = X.toarray()  # I am passing in a scipy sparse matrix

	# mask
	mask = np.sign(X)
	rows, columns = X.shape
	A = np.random.rand(rows, latent_features)
	A = np.maximum(A, eps)

	Y = linalg.lstsq(A, X)[0]
	Y = np.maximum(Y, eps)

	masked_X = mask * X
	X_est_prev = dot(A, Y)
	for i in range(1, max_iter + 1):
		top = dot(masked_X, Y.T)
		bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
		A *= top / bottom

		A = np.maximum(A, eps)
		top = dot(A.T, masked_X)
		bottom = dot(A.T, mask * dot(A, Y)) + eps
		Y *= top / bottom
		Y = np.maximum(Y, eps)
		if i % 5 == 0 or i == 1 or i == max_iter:
			print 'Iteration {}:'.format(i),
			X_est = dot(A, Y)
			err = mask * (X_est_prev - X_est)
			fit_residual = np.sqrt(np.sum(err ** 2))
			X_est_prev = X_est
			curRes = linalg.norm(mask * (X - X_est), ord='fro')
			print 'fit residual', np.round(fit_residual, 4),
			print 'total residual', np.round(curRes, 4)
			if curRes < error_limit or fit_residual < fit_error_limit:
				break
	return A, Y


if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
	
    R = np.array(R)
    print R

    nP, nQ = nmf(R, 2)
    print np.dot(nP, nQ)
	# To restore R, execute numpy.dot(np, nQ.T)

