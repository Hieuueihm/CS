import numpy as np
def cs_ista(y, A, lambda_=2e-5, epsilon=1e-4, itermax = 1000):
	N = A.shape[1]
	errors = []
	x_1 = np.zeros((N, 1))

	for i in range(itermax):
		g_1 = A.T@(y - A@x_1)
		alpha = 1

		x_2 = x_1 + alpha*g_1
		x_hat = np.sign(x_2) * np.maximum(np.abs(x_2) - alpha*lambda_, 0)
		err1 = np.linalg.norm(x_hat - x_1) / (np.linalg.norm(x_hat) + 1e-12)
		err2 = np.linalg.norm(y - A@x_hat)
		errors.append([err1, err2])
		if err1 < epsilon or err2 < epsilon:
			break
		else:
			x_1 = x_hat
	return x_hat, np.array(errors)