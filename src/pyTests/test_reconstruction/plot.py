from glob import glob
import numpy as np
import matplotlib.pyplot as plt

for f_name in glob("errors.*.npy"):
	errs = np.load(f_name)
	plt.figure()
	title = " ".join(f_name.split(".")[1:-1])
	plt.title(title)
	plt.plot(errs[0], 'ro-', label="CM")
	plt.plot(errs[1], 'ko-', label="Ref")
	plt.legend()
	plt.tight_layout()
plt.show()
