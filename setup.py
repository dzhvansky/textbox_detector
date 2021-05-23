from setuptools import find_packages, setup

requirements = [x.strip() for x in open("requirements.txt")]
url = "https://"

setup(
    name="textbox_detector",
    version="1.0.0",
    packages=find_packages(),
    url=url,
    license="",
    author="dzhvansky",
    author_email="",
    description="",
    install_requires=requirements,
)
