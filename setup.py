import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setup(
	name="xerus",
	version="4.0",
	author_email="contact@libxerus.org",
	description="A general purpose tensor library",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://libxerus.org",
	packages=['xerus'],
	# package_data={'xerus': ['*.so']},
	package_data={'xerus': ['xerus.so']},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
		"Operating System :: POSIX :: Linux",
	],
	install_requires = [
		"numpy",
	],
)
